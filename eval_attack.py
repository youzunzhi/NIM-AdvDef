# import hf_env
# hf_env.set_env('202111')

import time
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from pathlib import Path
from argparse import ArgumentParser
import os
import hfai
from hfai.datasets import ImageNet
# from torch.nn.parallel import DistributedDataParallel as TorchDDP
from hfai.nn.parallel import DistributedDataParallel as TorchDDP

from torch.utils.data.distributed import DistributedSampler
import hfai.distributed as dist
# import torch.distributed as dist
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import model.mae
from model import ClassifierWithDefenseNIM, ClassifierWithDefenseMAE

from util.misc import init_dist, setup_logger, log_info
from util import mae_dataset

from my_torchattacks import FGSM, PGD, AutoAttack


def main(local_rank):
    # init dist
    rank, world_size = init_dist(local_rank)

    # fix the seed for reproducibility
    torch.manual_seed(123 + rank)
    np.random.seed(123 + rank)
    torch.backends.cudnn.benchmark = True

    parser = ArgumentParser(description="Evaluate denoisae under attack")
    parser.add_argument("--backbone", type=str, default='vit_base')
    parser.add_argument("--defense_framework", type=str, default='nim_mae')
    parser.add_argument("--defense_name", type=str, default='gamma_25_3')
    parser.add_argument("--defense_epochs", type=int, default=800)
    parser.add_argument("--finetune_exp", type=str, default='nim_mae_gamma_25_3_on_uniform_0_30_vit_base_800eps')
    parser.add_argument("--finetune_epochs", type=int, default=100)
    parser.add_argument('--attacks', nargs='+')
    parser.add_argument('--sigma_list', nargs='+', type=int)
    parser.add_argument("--no_denoising", action='store_true')

    parser.add_argument('--test_adv', action='store_true')
    parser.add_argument("--targeted", action='store_true')
    args = parser.parse_args()

    defense_framework = args.defense_framework
    defense_name = args.defense_name
    defense_epochs = args.defense_epochs
    backbone = args.backbone
    finetune_exp = args.finetune_exp
    finetune_epochs = args.finetune_epochs
    targeted = args.targeted
    test_adv = args.test_adv
    no_denoising = args.no_denoising

    sigma_list = args.sigma_list
    # sigma_list = list(range(-5, 105, 5))
    attack_name_list = args.attacks
    # attack_name_list = ['none', 'fgsm_4', 'pgd_4_10', 'aa_Linf_4']

    batch_size = 16
    if 'large' in backbone or 'huge' in backbone:
        batch_size = 10
    for atk in attack_name_list:
        if 'aa' in atk or 'ce' in atk:
            batch_size = 1

    root_dir = 'youzunzhi/workspaces/nim/'
    if 'mae' in defense_framework:
        classifier_name = f'{backbone}_patch{16 if "huge" not in backbone else 14}'
        classifier = model.mae.__dict__[classifier_name](num_classes=1000, global_pool='vit21k_official' not in finetune_exp)
        if not no_denoising:
            defense_model_name = f"{defense_framework}_{classifier_name}"
            defense_model = model.mae.__dict__[defense_model_name](norm_pix_loss="nonorm" not in defense_name)
        else:
            defense_model = None
    else:
        raise NotImplementedError

    if test_adv:
        classifier_path = os.path.join(root_dir, "output/adv_finetune/")
    else:
        classifier_path = os.path.join(root_dir, "output/finetune/")
    classifier_path = os.path.join(classifier_path, f"{finetune_exp}")
    save_path = Path(os.path.join(classifier_path, 'eval_defense'))
    save_path.mkdir(exist_ok=True, parents=True)
    classifier_path = os.path.join(classifier_path, f"epochs/{finetune_epochs-1:04d}.pt")
    if 'nim' in defense_framework:
        ClassifierWithDefense = ClassifierWithDefenseNIM
    else:
        ClassifierWithDefense = ClassifierWithDefenseMAE
    classifier = TorchDDP(classifier.cuda(), device_ids=[local_rank])
    classifier_ckpt = torch.load(classifier_path, map_location='cpu')
    try:
        classifier.module.load_state_dict(classifier_ckpt['classifier'])
    except KeyError:
        classifier.module.load_state_dict(classifier_ckpt['model'])
    classifier.eval()

    if not no_denoising:
        defense_model = TorchDDP(defense_model.cuda(), device_ids=[local_rank])
        if len(sigma_list) == 1 and sigma_list[0] < 0:
            pass
        else:
            pretrain_model_pth = os.path.join(root_dir, f"output/pretrain/{defense_framework}_{defense_name}_{backbone}_{defense_epochs}eps/epochs/{defense_epochs-1:04d}.pt")
            pretrain_model_ckpt = torch.load(pretrain_model_pth, map_location="cpu")
            defense_model.module.load_state_dict(pretrain_model_ckpt['model'])
            defense_model.eval()

    writer = None
    if rank == 0:
        print(f"save_path: {save_path}")
        setup_logger(save_path, 'log', rank)

    val_dataset = mae_dataset.build_dataset(is_train=False)
    val_datasampler = DistributedSampler(val_dataset, shuffle=True)
    val_dataloader = val_dataset.loader(batch_size, sampler=val_datasampler, num_workers=8, pin_memory=True)

    start_sigma, start_step = -10, 0
    correct1, correct5, total = 0, 0, 0

    if (save_path / 'latest.pt').exists():
        try:
            ckpt = torch.load(save_path / 'latest.pt', map_location='cpu')
            start_sigma, start_step, start_attack_idx = ckpt['sigma'], ckpt['step'], ckpt['attack_idx']
            correct1, correct5, total = ckpt['correct1'], ckpt['correct5'], ckpt['total']
        except:
            pass

    max_sigma, sigma_step = 305, 5
    if 'nim' not in defense_framework:
        max_sigma = 105
    start_attack_idx = 0
    for sigma in range(start_sigma, max_sigma, sigma_step):
        if sigma not in sigma_list:
            continue
        if 'nim' not in defense_framework:
            sigma /= 100
        classifier_with_defense = ClassifierWithDefense(defense_model, classifier, sigma, no_denoising=no_denoising)
        classifier_with_defense.eval()

        for attack_idx in range(start_attack_idx, len(attack_name_list)):
            attack_name = attack_name_list[attack_idx]
            if 'none' in attack_name or 'fgsm' in attack_name:
                num_eval_img = 50000
            elif 'pgd' in attack_name:
                num_eval_img = 50000
            elif 'aa' in attack_name:
                num_eval_img = 5000
            else:
                raise NotImplementedError

            val_dataloader.set_step(start_step)
            if attack_name != 'none':
                attack_model = ClassifierWithDefense(defense_model, classifier, sigma,
                                                     no_defense_model='gb' in attack_name, no_denoising=no_denoising)
                attack_model.eval()
                if 'fgsm' in attack_name:
                    attack = FGSM(attack_model, eps=float(attack_name.split('_')[-1]) / 255)
                elif 'pgd' in attack_name:
                    attack = PGD(attack_model, eps=float(attack_name.split('_')[-2]) / 255,
                                steps=int(attack_name.split('_')[-1]))
                elif 'aa' in attack_name:
                    attack = AutoAttack(attack_model, norm='Linf',
                                        eps=int(attack_name.split('_')[-1]) / 255, version=attack_name.split('_')[-2], n_classes=1000)
                else:
                    raise NotImplementedError

                if targeted:
                    attack.set_mode_targeted_random()
                attack.set_normalization_used(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

            for step, batch in enumerate(val_dataloader):
                if total >= num_eval_img:
                    break
                step += start_step
                imgs, labels = [x.cuda(non_blocking=True) for x in batch]
                if attack_name != 'none':
                    imgs = attack(imgs, labels)

                with torch.no_grad():
                    outputs = classifier_with_defense(imgs)
                _, preds = outputs.topk(5, -1, True, True)

                correct1_batch = torch.eq(preds[:, :1], labels.unsqueeze(1)).sum()
                correct5_batch = torch.eq(preds, labels.unsqueeze(1)).sum()
                bs = torch.tensor(imgs.shape[0]).to(imgs.device)

                dist.all_reduce(correct1_batch)
                dist.all_reduce(correct5_batch)
                dist.all_reduce(bs)
                correct1 += correct1_batch.item()
                correct5 += correct5_batch.item()
                total += bs.item()

                if (('cw' in attack_name or 'aa' in attack_name) or ('pgd' in attack_name and step % 5 == 0)) and rank == 0:
                    log_info(f"{attack_name} {sigma} Step {step}: correct1: {correct1}, correct5: {correct5}, total: {total}")

                if rank == 0 and hfai.client.receive_suspend_command():
                    state = {
                        'sigma': sigma if 'nim' in defense_framework else sigma*100,
                        'step': step,
                        'attack_idx': attack_idx,
                        'correct1': correct1,
                        'correct5': correct5,
                        'total': total
                    }
                    torch.save(state, save_path / 'latest.pt', _use_new_zipfile_serialization=False)
                    time.sleep(5)
                    hfai.client.go_suspend()

            if rank == 0:
                acc1 = 100 * correct1 / total
                acc5 = 100 * correct5 / total
                log_info(
                    f"{finetune_exp} under {attack_name} "
                    f"{'(targeted) ' if targeted else ''}"
                    f"with sigma={sigma}: Acc1: {correct1}/{total}={acc1:.2f}%, Acc5: {correct5}/{total}={acc5:.2f}%)")
            correct1, correct5, total = 0, 0, 0
            start_step = 0

    if writer:
        writer.close()

    dist.barrier()
    classifier.reducer.stop()
    defense_model.reducer.stop()


if __name__ == "__main__":
    ngpus = torch.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=True)