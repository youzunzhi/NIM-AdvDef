import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import time
import datetime
from pathlib import Path
import numpy as np
import torch
from argparse import ArgumentParser
from tensorboardX import SummaryWriter

import hfai
import hfai.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as TorchDDP
from hfai.nn.parallel import DistributedDataParallel as HfaiDDP
from torch.utils.data.distributed import DistributedSampler

# import timm
# assert timm.__version__ == "0.3.2"  # version check
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import trunc_normal_

import model.mae
from util.misc import init_dist
from util.defense import get_noisy_imgs
from util import mae_lr_decay, mae_lr_sched, mae_pos_embed, mae_dataset

writer = None


def main(local_rank):

    # ----- experiments -----
    # init dist
    rank, world_size = init_dist(local_rank)

    # fix the seed for reproducibility
    torch.manual_seed(123 + rank)
    np.random.seed(123 + rank)
    torch.backends.cudnn.benchmark = True

    parser = ArgumentParser(description="Finetune with x_rec")
    parser.add_argument("--framework", type=str, default='nim_mae', choices=['mae', 'nim_mae'])
    parser.add_argument("--backbone", type=str, default='vit_base')
    parser.add_argument("--pretrain_exp", type=str, default='gamma_25_3')
    parser.add_argument("--pretrain_epochs", type=int, default=800)
    parser.add_argument("--on_original", action='store_true')
    parser.add_argument("--sigma_dist", type=str, default='uniform', choices=['global', 'gamma', 'uniform'])
    parser.add_argument("--global_sigma", type=float, default=0)
    parser.add_argument("--gamma_concentration", type=float, default=40)
    parser.add_argument("--gamma_scale", type=float, default=2)
    parser.add_argument("--uniform_low", type=float, default=0)
    parser.add_argument("--uniform_high", type=float, default=30)
    parser.add_argument("--no_denoising", action='store_true')
    parser.add_argument("--save_freq", type=int, default=5)

    args = parser.parse_args()

    backbone = args.backbone
    framework = args.framework
    pretrain_exp = args.pretrain_exp
    pretrain_epochs = args.pretrain_epochs
    on_original = args.on_original
    sigma_dist = args.sigma_dist
    global_sigma = args.global_sigma if args.global_sigma % 1 != 0 else int(args.global_sigma)
    gamma_concentration = args.gamma_concentration if args.gamma_concentration % 1 != 0 else int(
        args.gamma_concentration)
    gamma_scale = args.gamma_scale if args.gamma_scale % 1 != 0 else int(args.gamma_scale)
    uniform_low = args.uniform_low if args.uniform_low % 1 != 0 else int(args.uniform_low)
    uniform_high = args.uniform_high if args.uniform_high % 1 != 0 else int(args.uniform_high)
    no_denoising = args.no_denoising
    save_freq = args.save_freq

    if 'base' in backbone:
        epochs = 100
    elif 'large' in backbone:
        epochs = 50
    elif 'huge' in backbone:
        epochs = 50
    else:
        raise NotImplementedError

    exp_name = f"{pretrain_exp}"
    if not on_original:
        exp_name += f"_on_{sigma_dist}"
        if sigma_dist == 'global':
            exp_name += f"_{global_sigma}"
            eval_sigma = global_sigma
        elif sigma_dist == 'gamma':
            exp_name += f"_{gamma_concentration}_{gamma_scale}"
            eval_sigma = 70
        elif sigma_dist == 'uniform':
            exp_name += f"_{uniform_low}_{uniform_high}"
            eval_sigma = 70
        if no_denoising:
            exp_name += f"_no_denoising"
    else:
        exp_name += "_on_original"
        eval_sigma = -10

    save_path = Path(f"output/finetune/{framework}_{exp_name}_{backbone}_{pretrain_epochs}eps")
    save_path.mkdir(exist_ok=True, parents=True)
    (save_path / 'epochs').mkdir(exist_ok=True, parents=True)
    log_path = save_path / "runs"

    if rank == 0:
        print(f"save_path: {save_path}")
        global writer
        writer = SummaryWriter(log_path)

    # ----------------------

    # ----- data -----
    if 'mae' in framework:
        # hyper parameters
        batch_size = 16  # use 8 nodes, batch size 1024
        eff_batch_size = batch_size * world_size
        assert eff_batch_size == 1024

        train_dataset = mae_dataset.build_dataset(is_train=True)
        val_dataset = mae_dataset.build_dataset(is_train=False)
        train_datasampler = DistributedSampler(train_dataset, shuffle=True)
        train_dataloader = train_dataset.loader(batch_size, sampler=train_datasampler, num_workers=8, pin_memory=True,
                                                drop_last=True)

        val_datasampler = DistributedSampler(val_dataset, shuffle=False)
        val_dataloader = val_dataset.loader(batch_size, sampler=val_datasampler, num_workers=8, pin_memory=True)
    else:
        raise NotImplementedError

    # mixup two images and labels
    mixup = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=0.1,
        num_classes=1000,
    )
    # -----------------

    # ----- model -----
    pretrain_model_path = f"output/pretrain/{framework}_{pretrain_exp}_{backbone}_{pretrain_epochs}eps/epochs/{pretrain_epochs - 1:04d}.pt"
    pretrain_model_ckpt = torch.load(pretrain_model_path, map_location="cpu")
    if 'mae' in framework:
        drop_path = 0.2 if 'huge' in backbone else 0.1
        classifier_name = f'{backbone}_patch{16 if "huge" not in backbone else 14}'
        defense_model_name = f"{framework}_{classifier_name}"
        classifier = model.mae.__dict__[classifier_name](num_classes=1000, drop_path_rate=drop_path, global_pool=True)
        if not (save_path / 'latest.pt').exists():
            pretrain_model_ckpt_model = pretrain_model_ckpt['model']
            state_dict = classifier.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in pretrain_model_ckpt_model and pretrain_model_ckpt_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del pretrain_model_ckpt_model[k]

            # interpolate position embedding
            mae_pos_embed.interpolate_pos_embed(classifier, pretrain_model_ckpt_model)
            msg = classifier.load_state_dict(pretrain_model_ckpt_model, strict=False)

            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            # manually initialize fc layer
            trunc_normal_(classifier.head.weight, std=2e-5)

        classifier = hfai.nn.to_hfai(classifier)
        classifier = HfaiDDP(classifier.cuda(), device_ids=[local_rank])
    else:
        raise NotImplementedError
    # -----------------

    # ----- optimizer -----
    if 'mae' in framework:
        base_lr = 1e-3
        weight_decay = 0.05
        layer_decay = 0.75
        lr = base_lr * eff_batch_size / 256
        param_groups = mae_lr_decay.param_groups_lrd(classifier.module, weight_decay,
                                                     no_weight_decay_list=classifier.module.no_weight_decay(),
                                                     layer_decay=layer_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=lr)
        warmup_epochs, min_lr = 5, 1e-6
    else:
        raise NotImplementedError
    criterion = SoftTargetCrossEntropy()
    # ---------------------

    # --- resume from latest.pt ---
    start_epoch, start_step = 0, 0

    if (save_path / 'latest.pt').exists():
        try:
            ckpt = torch.load(save_path / 'latest.pt', map_location='cpu')
        except:
            ckpt = None
            for i in range(epochs-1, 0, -1):
                if (save_path / f'epochs/{i:04d}.pt').exists():
                    ckpt = torch.load(save_path / f'epochs/{i:04d}.pt', map_location='cpu')
                    break
        classifier.module.load_state_dict(ckpt['classifier'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch, start_step = ckpt['epoch'], ckpt['step']
    # -----------------------------

    # load defense_model for reconstructed image
    if not on_original and not no_denoising:
        defense_model = model.mae.__dict__[defense_model_name](norm_pix_loss="nonorm" not in pretrain_exp,
                                                               all_patch_loss='all' in pretrain_exp)
        defense_model.load_state_dict(pretrain_model_ckpt['model'])
        defense_model = hfai.nn.to_hfai(defense_model)
        defense_model = HfaiDDP(defense_model.cuda(), device_ids=[local_rank])
        defense_model.eval()

    # for clamp
    imagenet_std_reshape = torch.tensor(IMAGENET_DEFAULT_STD).view(-1, 1, 1).to(classifier.device)
    imagenet_mean_reshape = torch.tensor(IMAGENET_DEFAULT_MEAN).view(-1, 1, 1).to(classifier.device)

    # ----- train-then-validate ------
    for epoch in range(start_epoch, epochs):
        if start_step < len(train_dataloader):
            # -------- train --------
            classifier.train()
            # resume from epoch and step
            train_datasampler.set_epoch(epoch)
            train_dataloader.set_step(start_step)
            steps_per_epoch = len(train_dataloader) + start_step
            for step, batch in enumerate(train_dataloader):
                step += start_step
                # we use a per iteration (instead of per epoch) lr scheduler
                mae_lr_sched.adjust_learning_rate(optimizer, step / steps_per_epoch + epoch, warmup_epochs, lr, min_lr,
                                                  epochs)

                imgs, labels = [x.cuda(non_blocking=True) for x in batch]
                imgs, labels = mixup(imgs, labels)

                if not on_original:
                    noisy_imgs = get_noisy_imgs(sigma_dist, imgs, global_sigma, gamma_concentration,
                                                gamma_scale, uniform_low, uniform_high)

                    if not no_denoising:
                        with torch.no_grad():
                            imgs, _ = defense_model(imgs, noisy_imgs)
                    else:
                        imgs = noisy_imgs
                    # clamp
                    imgs = imgs * imagenet_std_reshape + imagenet_mean_reshape
                    imgs = imgs.clamp(0, 1)
                    imgs = (imgs - imagenet_mean_reshape) / imagenet_std_reshape

                outputs = classifier(imgs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # log
                dist.all_reduce(loss)
                loss = loss.item() / dist.get_world_size()
                global_steps = epoch * steps_per_epoch + step
                if rank == 0 and step % 50 == 0:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    writer.add_scalar("lr", cur_lr, global_steps)
                    writer.add_scalar("loss", loss, global_steps)
                    print(f"{(datetime.datetime.now()).strftime('%m-%d %H:%M:%S')} "
                          f"Epoch: {epoch}, Step: {step}, Loss: {loss:.3f}, lr: {cur_lr:.6f}", flush=True)

                if rank == 0 and hfai.client.receive_suspend_command():
                    state = {
                        'classifier': classifier.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'step': step + 1
                    }
                    torch.save(state, save_path / 'latest.pt', _use_new_zipfile_serialization=False)
                    time.sleep(5)
                    hfai.client.go_suspend()
            # -----------------------
        # -------- eval --------
        classifier.eval()
        with torch.no_grad():
            loss, correct1, correct5, total = torch.zeros(4).cuda()
            for step, batch in enumerate(val_dataloader):
                imgs, labels = [x.cuda(non_blocking=True) for x in batch]
                if eval_sigma >= 0:
                    noisy_imgs = get_noisy_imgs('global', imgs, eval_sigma)
                    if not no_denoising:
                        imgs, _ = defense_model(imgs, noisy_imgs)
                    else:
                        imgs = noisy_imgs
                    # clamp
                    imgs = imgs * imagenet_std_reshape + imagenet_mean_reshape
                    imgs = imgs.clamp(0, 1)
                    imgs = (imgs - imagenet_mean_reshape) / imagenet_std_reshape
                outputs = classifier(imgs)
                _, preds = outputs.topk(5, -1, True, True)
                correct1 += torch.eq(preds[:, :1], labels.unsqueeze(1)).sum()
                correct5 += torch.eq(preds, labels.unsqueeze(1)).sum()
                total += imgs.size(0)

            for x in [correct1, correct5, total]:
                dist.all_reduce(x)

            if rank == 0:
                acc1 = 100 * correct1.item() / total.item()
                acc5 = 100 * correct5.item() / total.item()
                print(f"{(datetime.datetime.now()).strftime('%m-%d %H:%M:%S')} Epoch {epoch}: [Final] Sigma={eval_sigma}: correct1: {correct1.item()}, correct5: "
                      f"{correct5.item()}, total: {total.item()}, Acc1: {acc1:.2f}%, Acc5: {acc5:.2f}%", flush=True)

                state = {
                    'classifier': classifier.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch+1,
                    'step': 0
                }
                if epoch % save_freq == save_freq - 1:
                    torch.save(state, save_path / f'epochs/{epoch:04d}.pt', _use_new_zipfile_serialization=False)

        start_step = 0  # reset
    # --------------------------------

    dist.barrier()
    classifier.reducer.stop()
    defense_model.reducer.stop()

    if writer:
        writer.close()


if __name__ == "__main__":
    ngpus = torch.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=True)
