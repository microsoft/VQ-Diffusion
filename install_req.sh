#!/bin/bash
pip install torch==1.9.0 torchvision --no-cache-dir -U | cat
pip install omegaconf pytorch-lightning --no-cache-dir -U | cat
pip install timm==0.3.4 --no-cache-dir -U | cat
pip install tensorboard==1.15.0 --no-cache-dir -U | cat
pip install lmdb tqdm --no-cache-dir -U | cat
pip install einops ftfy --no-cache-dir -U | cat
pip install git+https://github.com/openai/DALL-E.git --no-cache-dir -U | cat