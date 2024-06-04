# NIM-AdvDef
Code repo for the paper "[Beyond Pretrained Features: Noisy Image Modeling Provides Adversarial Defense](https://arxiv.org/pdf/2302.01056)"

So far the code runs on [HAI Platform](https://hfailab.github.io/hai-platform/). Please make your own modifications to run it on your local machine, and feel free to contact me if you have any questions.

## Requirements
- Python 3.8
- PyTorch 1.10.2
- timm 0.4.12

## Usage
### Pretraining
To train an NIM-MAE pretrained model with $\sigma \sim \Gamma(25,3)$, run the following command:
```bash
python pretrain.py --framework nim_mae --sigma_dist gamma --gamma_concentration 25 --gamma_scale 3
```
You can also pretrain a MAE baseline counterpart ($\gamma = 0.75$) by running:
```bash
python pretrain.py --framework mae --sigma_dist global --global_sigma 0.75
```

### Fine-tuning
To fine-tune a pretrained model with denoised images by the pretrained model, run:
```bash
python finetune.py --framework nim_mae --pretrain_exp gamma_25_3 --sigma_dist uniform --uniform_low 0 --uniform_high 30
```

### Evaluation
To evaluate the adversarial robustness of a fine-tuned model using the pretrained model as a defense, run:
```bash
python eval_attack.py --sigma_list -5 70 140 --attacks none fgsm_4 pgd_4_10
```

## Acknowledgements
- This repo is based on the [MAE repo](https://github.com/facebookresearch/mae/tree/main). 
- The robustness evaluation code is copied from [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch).

## Citation
If you find this code useful, please consider citing:
```
@inproceedings{you2023beyond,
    title={Beyond Pretrained Features: Noisy Image Modeling Provides Adversarial Defense},
    author={Zunzhi You and Daochang Liu and Bohyung Han and Chang Xu},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=niHkj9ixUZ}
}
```
