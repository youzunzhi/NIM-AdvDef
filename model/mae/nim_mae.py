from functools import partial

import torch
import torch.nn as nn

# from timm.models.vision_transformer import PatchEmbed, Block
from .mae import MaskedAutoencoderViT
from .utils import get_2d_sincos_pos_embed

class NIM_MAE_ViT(MaskedAutoencoderViT):

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            target_mean = target.mean(dim=-1, keepdim=True)
            target_var = target.var(dim=-1, keepdim=True)
            target_std = (target_var + 1.e-6) ** .5
            target = (target - target_mean) / target_std

        loss = (pred - target) ** 2
        loss = loss.mean()  # [N, L], mean loss per patch

        # unnormalize and unpatchify to image-like prediction
        if self.norm_pix_loss:
            pred = pred * target_std + target_mean
        pred = self.unpatchify(pred)

        return loss, pred

    def forward(self, imgs, noisy_imgs):
        latent = self.forward_encoder(noisy_imgs)
        pred = self.forward_decoder(latent)
        loss, pred = self.forward_loss(imgs, pred)

        return pred, loss


def nim_mae_vit_base_patch16_dec512d8b(**kwargs):
    model = NIM_MAE_ViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def nim_mae_vit_large_patch16_dec512d8b(**kwargs):
    model = NIM_MAE_ViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def nim_mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = NIM_MAE_ViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
nim_mae_vit_base_patch16 = nim_mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
nim_mae_vit_large_patch16 = nim_mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
nim_mae_vit_huge_patch14 = nim_mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

