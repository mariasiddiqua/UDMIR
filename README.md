# UDMIR
UDMIR: Towards Unified Marine Image Restoration Using Dual-Attention Transformer Diffusion

## Prerequisites
torch>=1.6, torchvision, numpy, pandas, tqdm, lmdb, opencv-python, pillow, tensorboardx, wandb

## Getting Started
- Clone this repo:
```bash
git clone https://github.com/mariasiddiqua/UDMIR/
cd udmir
```
- Dataset Download: [Google Drive](https://drive.google.com/file/d/1xFDsCrFx5WQLWPgVkh5coM70Dgh1AFNq/view?usp=sharing)

- Train a model:

```bash
python train.py -c config/uw_train.json
```

- Test the model:

```bash
python train.py -c config/uw_test.json
```
