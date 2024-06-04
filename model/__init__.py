import torch.nn as nn
import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from util import get_noisy_imgs


class ClassifierWithDefense(nn.Module):
    def __init__(self, defense_model, classifier, defense_sigma, no_defense_model=False, no_denoising=False):
        super().__init__()
        self.defense_model = defense_model
        self.classifier = classifier
        self.defense_sigma = defense_sigma
        if defense_sigma < 0:
            self.no_defense_model = True
        else:
            self.no_defense_model = no_defense_model
        self.no_denoising = no_denoising

        # try:
        #     self.clamp_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(-1, 1, 1).to(classifier.device)
        #     self.clamp_std = torch.tensor(IMAGENET_DEFAULT_STD).view(-1, 1, 1).to(classifier.device)
        # except:
        #     self.clamp_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(-1, 1, 1)
        #     self.clamp_std = torch.tensor(IMAGENET_DEFAULT_STD).view(-1, 1, 1)
        self.clamp_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(-1, 1, 1)
        self.clamp_std = torch.tensor(IMAGENET_DEFAULT_STD).view(-1, 1, 1)

    def forward(self, imgs):
        raise NotImplementedError


class ClassifierWithDefenseNIM(ClassifierWithDefense):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, imgs):
        if not self.no_defense_model:
            noisy_imgs = get_noisy_imgs('global', imgs, global_sigma=self.defense_sigma)
            if not self.no_denoising:
                imgs, loss = self.defense_model(imgs, noisy_imgs)
            else:
                imgs = noisy_imgs
            try:
                imgs = imgs * self.clamp_std + self.clamp_mean
            except:
                self.clamp_std = self.clamp_std.to(imgs.device)
                self.clamp_mean = self.clamp_mean.to(imgs.device)
                imgs = imgs * self.clamp_std + self.clamp_mean
            imgs = imgs.clamp(0, 1)
            imgs = (imgs - self.clamp_mean) / self.clamp_std

            self.dgr_img = noisy_imgs[0]
            self.rec_img = imgs[0]

        outputs = self.classifier(imgs)

        return outputs


class ClassifierWithDefenseMAE(ClassifierWithDefense):
    def forward(self, imgs):
        if not self.no_defense_model:
            preds, mask, _ = self.defense_model(imgs, self.defense_sigma)

            if mask is not None:
                try:
                    mask_imgs = self.defense_model.encoder.patch_embed.unpatchify(mask.unsqueeze(-1).repeat(1, 1, 768))
                except:
                    mask_imgs = self.defense_model.module.encoder.patch_embed.unpatchify(mask.unsqueeze(-1).repeat(1, 1, 768))

                mask_imgs = imgs * (1 - mask_imgs)
                self.dgr_img = mask_imgs[0]
            else:
                self.dgr_img = imgs[0]

            preds = preds * self.clamp_std + self.clamp_mean
            preds = preds.clamp(0, 1)
            imgs = (preds - self.clamp_mean) / self.clamp_std
            self.rec_img = imgs[0]

        outputs = self.classifier(imgs)

        return outputs
