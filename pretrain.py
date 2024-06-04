import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import time
import datetime
from pathlib import Path
import numpy as np
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
import torch
from torchvision import transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import timm.optim.optim_factory as optim_factory

import hfai
import hfai.distributed as dist
from hfai.datasets import ImageNet
# from torch.nn.parallel import DistributedDataParallel as TorchDDP
from hfai.nn.parallel import DistributedDataParallel as HfaiDDP
from torch.utils.data.distributed import DistributedSampler

import model.mae
from util.misc import init_dist
from util.defense import get_noisy_imgs
from util import mae_lr_sched

writer = None


def main(local_rank):
    # init dist
    rank, world_size = init_dist(local_rank)

    # fix the seed for reproducibility
    torch.manual_seed(123 + rank)
    np.random.seed(123 + rank)
    torch.backends.cudnn.benchmark = True

    parser = ArgumentParser(description="Pretrain")
    parser.add_argument("--framework", type=str, default='nim_mae')
    parser.add_argument("--backbone", type=str, default='vit_base')
    parser.add_argument("--pretrain_epochs", type=int, default=800)
    parser.add_argument("--sigma_dist", type=str, default='gamma', choices=['global', 'gamma', 'uniform'])
    parser.add_argument("--global_sigma", type=float, default=0)
    parser.add_argument("--gamma_concentration", type=float, default=25)
    parser.add_argument("--gamma_scale", type=float, default=3)
    parser.add_argument("--uniform_low", type=float, default=0)
    parser.add_argument("--uniform_high", type=float, default=150)
    parser.add_argument("--nonorm", action='store_true')
    parser.add_argument("--all_patch_loss", action='store_true')
    args = parser.parse_args()

    backbone = args.backbone
    framework = args.framework
    pretrain_epochs = args.pretrain_epochs
    sigma_dist = args.sigma_dist
    global_sigma = args.global_sigma if args.global_sigma % 1 != 0 else int(args.global_sigma)
    gamma_concentration = args.gamma_concentration if args.gamma_concentration % 1 != 0 else int(args.gamma_concentration)
    gamma_scale = args.gamma_scale if args.gamma_scale % 1 != 0 else int(args.gamma_scale)
    uniform_low = args.uniform_low if args.uniform_low % 1 != 0 else int(args.uniform_low)
    uniform_high = args.uniform_high if args.uniform_high % 1 != 0 else int(args.uniform_high)
    nonorm = args.nonorm
    all_patch_loss = args.all_patch_loss
    if framework == 'mae':
        assert sigma_dist == 'global' and 0 <= global_sigma <= 1

    exp_name = f"{sigma_dist}"
    if sigma_dist == 'global':
        exp_name += f"_{global_sigma}"
    elif sigma_dist == 'gamma':
        exp_name += f"_{gamma_concentration}_{gamma_scale}"
    elif sigma_dist == 'uniform':
        exp_name += f"_{uniform_low}_{uniform_high}"
    if nonorm:
        exp_name += f"_nonorm"
    save_path = Path(f"output/pretrain/{framework}_{exp_name}_{backbone}_{pretrain_epochs}eps")
    save_path.mkdir(exist_ok=True, parents=True)
    (save_path / 'epochs').mkdir(exist_ok=True, parents=True)
    log_path = save_path / "runs"

    if rank == 0 and local_rank == 0:
        print(f"save_path: {save_path}")
        global writer
        writer = SummaryWriter(log_path)

    if 'mae' in framework:
        # hyper parameters
        if 'base' in backbone:
            batch_size = 64
            accum_iter = 1
        elif 'large' in backbone:
            batch_size = 32
            accum_iter = 2
        elif 'huge' in backbone:
            batch_size = 16
            accum_iter = 4  # node == 8
        else:
            raise NotImplementedError
        total_batch_size = batch_size * world_size * accum_iter
        assert total_batch_size == 4096
        base_lr = 1.5e-4
        lr = base_lr * total_batch_size / 256
        weight_decay = 0.05
        min_lr = 0
        warmup_epochs = 40
        clip_grad = None
        # ----- data -----
        mode = transforms.InterpolationMode.BICUBIC
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=mode),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])  # train transform

        dataset = ImageNet(split="train", transform=train_transform)
        sampler = DistributedSampler(dataset, shuffle=True)
        dataloader = dataset.loader(batch_size, sampler=sampler, num_workers=8, pin_memory=True)
        # ----------------
        # ----- model -----
        pretrain_model = model.mae.__dict__[f'{framework}_{backbone}_patch{16 if "huge" not in backbone else 14}'](norm_pix_loss=not nonorm, all_patch_loss=all_patch_loss)
        pretrain_model = hfai.nn.to_hfai(pretrain_model, verbose=False)
        pretrain_model = HfaiDDP(pretrain_model.cuda(), device_ids=[local_rank])

        param_groups = optim_factory.add_weight_decay(pretrain_model.module, weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
        # --------------------
    else:
        raise NotImplementedError

    # --- resume from latest.pt ---
    start_epoch, start_step = 0, 0
    if (save_path / 'latest.pt').exists():
        try:
            ckpt = torch.load(save_path / 'latest.pt', map_location='cpu')
        except:
            ckpt = None
            for i in range(pretrain_epochs-1, 0, -1):
                if (save_path / f'epochs/{i:04d}.pt').exists():
                    ckpt = torch.load(save_path / f'epochs/{i:04d}.pt', map_location='cpu')
                    break
        pretrain_model.module.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch, start_step = ckpt['epoch'], ckpt['step']
    # -----------------------------

    # -------- pretrain --------
    for epoch in range(start_epoch, pretrain_epochs):
        pretrain_model.train()
        # resume from epoch and step
        sampler.set_epoch(epoch)
        dataloader.set_step(start_step)
        steps_per_epoch = len(dataloader) + start_step
        for step, batch in enumerate(dataloader):
            step += start_step
            if 'mae' in framework:
                # we use a per iteration (instead of per epoch) lr scheduler
                if step % accum_iter == 0:
                    mae_lr_sched.adjust_learning_rate(optimizer, step / steps_per_epoch + epoch, warmup_epochs, lr, min_lr, pretrain_epochs)

            imgs = batch[0].cuda(non_blocking=True)
            if 'nim' in framework:
                noisy_imgs = get_noisy_imgs(sigma_dist, imgs,
                                            global_sigma, gamma_concentration, gamma_scale,
                                            uniform_low, uniform_high)
                preds, loss = pretrain_model(imgs, noisy_imgs)
            elif framework == 'mae':
                loss, _, _ = pretrain_model(imgs, global_sigma)
            else:
                raise NotImplementedError

            loss = loss / accum_iter

            loss.backward()

            if clip_grad:
                grad_norm = torch.nn.utils.clip_grad_norm_(pretrain_model.parameters(), clip_grad)
            # weights update
            if ((step + 1) % accum_iter == 0) or (step + 1 == steps_per_epoch):
                optimizer.step()
                optimizer.zero_grad()

            # log
            dist.all_reduce(loss)
            loss = loss.item() / dist.get_world_size()
            global_steps = epoch * steps_per_epoch + step
            if rank == 0 and step % 50 == 0:
                cur_lr =  optimizer.param_groups[0]["lr"]
                writer.add_scalar("lr", cur_lr, global_steps)
                writer.add_scalar("loss", loss, global_steps)
                print(f"{(datetime.datetime.now()).strftime('%m-%d %H:%M:%S')} Epoch: {epoch}, Step: {step}, Loss: {loss:.3f}, lr: {cur_lr:.6f}", flush=True)

            # save checkpoint if going to suspend
            if rank == 0 and hfai.client.receive_suspend_command():
                if epoch > start_epoch or step - start_step >= 8:
                    state = {
                        'model': pretrain_model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch if step < steps_per_epoch-1 else epoch+1,
                        'step': step + 1 if step < steps_per_epoch-1 else 0
                    }
                    torch.save(state, save_path / 'latest.pt', _use_new_zipfile_serialization=False)
                time.sleep(5)
                hfai.client.go_suspend()

        start_step = 0  # reset

        # save
        if rank == 0 and epoch % 10 == 9:
            state = {
                'model': pretrain_model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'step': 0
            }
            torch.save(state, save_path / f'epochs/{epoch:04d}.pt', _use_new_zipfile_serialization=False)
    # --------------------------

    dist.barrier()
    pretrain_model.reducer.stop()

    if writer:
        writer.close()


if __name__ == "__main__":
    ngpus = torch.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=True)