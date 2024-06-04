import torch
import numpy as np
from torchvision import transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def add_gaussian_noise(imgs, sigma, imagenet_std):
    noise = torch.randn_like(imgs).to(imgs.device) * sigma / 255
    noise = transforms.Normalize(mean=[0, 0, 0], std=imagenet_std)(noise)
    noisy_imgs = imgs + noise
    return noisy_imgs


def get_noisy_imgs(sigma_dist, imgs,
                   global_sigma=None, concentration=None, scale=None, low=None, high=None):
    if sigma_dist == 'global':
        assert global_sigma >= 0
        noisy_imgs = add_gaussian_noise(imgs, global_sigma, IMAGENET_DEFAULT_STD)
    elif sigma_dist == 'gamma':  # Gaussian noise whose sigma follows Gamma
        bs = len(imgs)
        sigma = torch.distributions.Gamma(concentration, 1/scale).sample([bs]).view(bs, 1, 1, 1).to(imgs.device)
        noisy_imgs = add_gaussian_noise(imgs, sigma, IMAGENET_DEFAULT_STD)
    elif sigma_dist == 'uniform':
        bs = len(imgs)
        sigma = torch.distributions.Uniform(low, high).sample([bs]).view(bs, 1, 1, 1).to(imgs.device)
        noisy_imgs = add_gaussian_noise(imgs, sigma, IMAGENET_DEFAULT_STD)
    else:
        raise Exception('wrong noise type')

    return noisy_imgs

class MaskGeneratorRandomRatio:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4,
                 mask_ratio_dist='global', global_val=0.6, uniform_low=0., uniform_high=1.):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        if mask_ratio_dist == 'global':
            self.mask_ratio = global_val
        elif mask_ratio_dist == 'uniform':
            self.mask_ratio = np.random.uniform(uniform_low, uniform_high)
        else:
            raise NotImplementedError
        assert self.mask_ratio <= 1

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask