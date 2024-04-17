# VQ-Diffusion (CVPR2022, Oral) and <br> Improved VQ-Diffusion

## Overview

This is the official repo for the paper: [Vector Quantized Diffusion Model for Text-to-Image Synthesis](https://arxiv.org/pdf/2111.14822.pdf) and [Improved Vector Quantized Diffusion Models](https://arxiv.org/pdf/2205.16007.pdf).

The code is the same as https://github.com/cientgu/VQ-Diffusion, some issues that have been raised can refer to it.

> VQ-Diffusion is based on a VQ-VAE whose latent space is modeled by a conditional variant of the recently developed Denoising Diffusion Probabilistic Model (DDPM). It produces significantly better text-to-image generation results when compared with Autoregressive models with similar numbers of parameters. Compared with previous GAN-based methods, VQ-Diffusion can handle more complex scenes and improve the synthesized image quality by a large margin.

## Framework

<img src='figures/framework.png' width='600'>

## Integration with 🤗 Diffusers library

VQ-Diffusion is now also available in 🧨 Diffusers and accesible via the [VQDiffusionPipeline](https://huggingface.co/docs/diffusers/api/pipelines/vq_diffusion).
Diffusers allows you to test VQ-Diffusion in just a couple lines of code.

You can install diffusers as follows:

```
pip install diffusers torch accelerate transformers
```

And then try out the model with just a couple lines of code:

```python
import torch
from diffusers import VQDiffusionPipeline

pipeline = VQDiffusionPipeline.from_pretrained("microsoft/vq-diffusion-ithq", torch_dtype=torch.float16, revision="fp16")
pipeline = pipeline.to("cuda")

image = pipeline("teddy bear playing in the pool").images[0]

# save image
image.save("./teddy_bear.png")
```

You can find the model card of the **ITHQ** checkpoint [here](https://huggingface.co/microsoft/vq-diffusion-ithq).

## Requirements

We suggest to use the [docker](https://hub.docker.com/layers/164588520/cientgu/pytorch1.9.0/latest/images/sha256-e4e8694817152b4d9295242044f2e0f7f35f41cf7055ab2942a768acc42c7858?context=repo). Also, you may run:
```
bash install_req.sh
```

## Data Preparing

### Microsoft COCO

```
│MSCOCO_Caption/
├──annotations/
│  ├── captions_train2014.json
│  ├── captions_val2014.json
├──train2014/
│  ├── train2014/
│  │   ├── COCO_train2014_000000000009.jpg
│  │   ├── ......
├──val2014/
│  ├── val2014/
│  │   ├── COCO_val2014_000000000042.jpg
│  │   ├── ......
```

### CUB-200

```
│CUB-200/
├──images/
│  ├── 001.Black_footed_Albatross/
│  ├── 002.Laysan_Albatross
│  ├── ......
├──text/
│  ├── text/
│  │   ├── 001.Black_footed_Albatross/
│  │   ├── 002.Laysan_Albatross
│  │   ├── ......
├──train/
│  ├── filenames.pickle
├──test/
│  ├── filenames.pickle
```

### ImageNet

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## Pretrained Model
We release four text-to-image pretrained model, trained on **Conceptual Caption**, **MSCOCO**, **CUB200**, and **LAION-human** datasets. Also, we release the **ImageNet** pretrained model, and provide the **CLIP** pretrained model for convenient. These should be put under OUTPUT/pretrained_model/ .
These pretrained model file may be large because they are training checkpoints, which contains gradient information, optimizer information, ema model and others.

Besides, we release four pretrained models with learnable classifier-free on **ITHQ**, **ImageNet**, **Conceptual Caption** and **MSCOCO** dataset.

We provide the VQVAE models on **FFHQ**, **OpenImages**, and **ImageNet** datasets, these models are from [Taming Transformer](https://github.com/CompVis/taming-transformers), we provide them here for convenient. Please put them under OUTPUT/pretrained_model/taming_dvae/ .

To support ITHQ dataset, we trained a new VQVAE model on **ITHQ** dataset.

For your convenience, we provide a script for downloading all models. You may run `bash vqdiffusion_download_checkpoints.sh`.

## Inference
To generate image from in-the-wild text:
```
from inference_VQ_Diffusion import VQ_Diffusion
VQ_Diffusion_model = VQ_Diffusion(config='configs/ithq.yaml', path='OUTPUT/pretrained_model/ithq_learnable.pth')

# Inference VQ-Diffusion
VQ_Diffusion_model.inference_generate_sample_with_condition("teddy bear playing in the pool", truncation_rate=0.86, save_root="RESULT", batch_size=4)

# Inference Improved VQ-Diffusion with learnable classifier-free sampling
VQ_Diffusion_model.inference_generate_sample_with_condition("teddy bear playing in the pool", truncation_rate=1.0, save_root="RESULT", batch_size=4, guidance_scale=5.0)
VQ_Diffusion_model.inference_generate_sample_with_condition("a long exposure photo of waterfall", truncation_rate=1.0, save_root="RESULT", batch_size=4, guidance_scale=5.0)

# Inference Improved VQ-Diffusion with fast/high-quality inference
VQ_Diffusion_model.inference_generate_sample_with_condition("a long exposure photo of waterfall", truncation_rate=0.86, save_root="RESULT", batch_size=4, infer_speed=0.5) # high-quality inference, 0.5x inference speed
VQ_Diffusion_model.inference_generate_sample_with_condition("a long exposure photo of waterfall", truncation_rate=0.86, save_root="RESULT", batch_size=4, infer_speed=2) # fast inference, 2x inference speed
# infer_speed shoule be float in [0.1, 10], larger infer_speed means faster inference and smaller infer_speed means slower inference

# Inference Improved VQ-Diffusion with purity sampling
VQ_Diffusion_model.inference_generate_sample_with_condition("a long exposure photo of waterfall", truncation_rate=0.86, save_root="RESULT", batch_size=4, prior_rule=2, prior_weight=1) # purity sampling

# Inference Improved VQ-Diffusion with both learnable classifier-free sampling and fast inference
VQ_Diffusion_model.inference_generate_sample_with_condition("a long exposure photo of waterfall", truncation_rate=1.0, save_root="RESULT", batch_size=4, guidance_scale=5.0, infer_speed=2) # classifier-free guidance and fast inference
```

To generate image from given text on MSCOCO/CUB/CC datasets:
```
from inference_VQ_Diffusion import VQ_Diffusion
VQ_Diffusion_model = VQ_Diffusion(config='OUTPUT/pretrained_model/config_text.yaml', path='OUTPUT/pretrained_model/coco_learnable.pth')

# Inference VQ-Diffusion
VQ_Diffusion_model.inference_generate_sample_with_condition("A group of elephants walking in muddy water", truncation_rate=0.86, save_root="RESULT", batch_size=4)

# Inference Improved VQ-Diffusion with learnable classifier-free sampling
VQ_Diffusion_model.inference_generate_sample_with_condition("A group of elephants walking in muddy water", truncation_rate=1.0, save_root="RESULT", batch_size=4, guidance_scale=3.0)
```
You may change coco_learnable.pth to other pretrained model to test different text.

To generate image from given ImageNet class label:
```
from inference_VQ_Diffusion import VQ_Diffusion

# Inference VQ-Diffusion
VQ_Diffusion_model = VQ_Diffusion(config='OUTPUT/pretrained_model/config_imagenet.yaml', path='OUTPUT/pretrained_model/imagenet_pretrained.pth')
VQ_Diffusion_model.inference_generate_sample_with_class(407, truncation_rate=0.86, save_root="RESULT", batch_size=4)


# Inference Improved VQ-Diffusion with classifier-free sampling
VQ_Diffusion_model = VQ_Diffusion(config='configs/imagenet.yaml', path='OUTPUT/pretrained_model/imagenet_learnable.pth', imagenet_cf=True)
VQ_Diffusion_model.inference_generate_sample_with_class(407, truncation_rate=0.94, save_root="RESULT", batch_size=8, guidance_scale=1.5)
```

## Training
First, change the data_root to correct path in configs/coco.yaml or other configs.

Train Text2Image generation on MSCOCO dataset:
```
python running_command/run_train_coco.py
```

Train Text2Image generation on CUB200 dataset:
```
python running_command/run_train_cub.py
```

Train conditional generation on ImageNet dataset:
```
python running_command/run_train_imagenet.py
```

Train unconditional generation on FFHQ dataset:
```
python running_command/run_train_ffhq.py
```

Fine-tune Text2Image generation on MSCOCO dataset with learnable classifier-free:
```
python running_command/run_tune_coco.py
```

## Cite VQ-Diffusion
if you find our code helpful for your research, please consider citing:
```
@article{gu2021vector,
  title={Vector Quantized Diffusion Model for Text-to-Image Synthesis},
  author={Gu, Shuyang and Chen, Dong and Bao, Jianmin and Wen, Fang and Zhang, Bo and Chen, Dongdong and Yuan, Lu and Guo, Baining},
  journal={arXiv preprint arXiv:2111.14822},
  year={2021}
}
```
## Acknowledgement
Thanks to everyone who makes their code and models available. In particular,

- [Multinomial Diffusion](https://github.com/ehoogeboom/multinomial_diffusion)
- [Taming Transformer](https://github.com/CompVis/taming-transformers)
- [Improved DDPM](https://github.com/openai/improved-diffusion)
- [Clip](https://github.com/openai/CLIP)

### License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information
For help or issues using VQ-Diffusion, please submit a GitHub issue.
For other communications related to VQ-Diffusion, please contact Shuyang Gu (gsy777@mail.ustc.edu.cn) or Dong Chen (doch@microsoft.com).
