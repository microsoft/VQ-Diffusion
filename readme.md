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

pipeline = VQDiffusionPipeline.from_pretrained("microsoft/vq-diffusion-ithq", torch_dtype=torch.float16)
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
We release four text-to-image pretrained model, trained on [Conceptual Caption](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/CC_pretrained.pth?sv=2021-10-04&st=2023-05-30T06%3A43%3A20Z&se=2030-05-31T06%3A43%3A00Z&sr=b&sp=r&sig=YtR4AADRLWFMf611J4M7bhXwDRQCYOwFs5gk4I7nii4%3D), [MSCOCO](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/coco_pretrained.pth?sv=2021-10-04&st=2023-05-30T06%3A43%3A38Z&se=2030-05-31T06%3A43%3A00Z&sr=b&sp=r&sig=YnXK5yNZffwL4FuGddxjqehevzf99Ie6CPcm8ACfmFM%3D), [CUB200](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/cub_pretrained.pth?sv=2021-10-04&st=2023-05-30T06%3A43%3A51Z&se=2030-05-31T06%3A43%3A00Z&sr=b&sp=r&sig=0VfK8vmU5J6BjYfezAmzDH13wlqpSrmwxXg01lnpFqc%3D), and [LAION-human](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/human_pretrained.pth?sv=2021-10-04&st=2023-05-30T06%3A44%3A03Z&se=2030-05-31T06%3A44%3A00Z&sr=b&sp=r&sig=uAIS2maOaHW5BewVXcU8ZjdhClpZud9Moj6aTwL8rto%3D) datasets. Also, we release the [ImageNet](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/imagenet_pretrained.pth?sv=2021-10-04&st=2023-05-30T06%3A44%3A15Z&se=2030-05-31T06%3A44%3A00Z&sr=b&sp=r&sig=%2BKhINCmnQHevAp3ew6WpucmePIHmW2UA%2Ff3M1UVxMDc%3D) pretrained model, and provide the [CLIP](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/ViT-B-32.pt?sv=2021-10-04&st=2023-05-30T06%3A44%3A27Z&se=2030-05-31T06%3A44%3A00Z&sr=b&sp=r&sig=7y5lmD6yhIZUJsDPN80XtMcs5ip%2FtPR0OrX1MrLCalY%3D) pretrained model for convenient. These should be put under OUTPUT/pretrained_model/ .
These pretrained model file may be large because they are training checkpoints, which contains gradient information, optimizer information, ema model and others.

Besides, we release four pretrained models with learnable classifier-free on [ITHQ](https://facevcstandard.blob.core.windows.net/t-zhitang/release/improved_vq_diffusion/ithq_learnable.pth?sv=2021-10-04&st=2023-05-30T06%3A32%3A34Z&se=2030-05-31T06%3A32%3A00Z&sr=b&sp=r&sig=eG0wyxLSPt6Y%2FGTuA2eVDI8rfIz5emZXLAGyeXuMtpA%3D), [ImageNet](https://facevcstandard.blob.core.windows.net/t-zhitang/release/improved_vq_diffusion/imagenet_learnable.pth?sv=2021-10-04&st=2023-05-30T06%3A33%3A19Z&se=2030-05-31T06%3A33%3A00Z&sr=b&sp=r&sig=OLzbazTlqa9N77bwcLhAfZjuT10jj0tVqCezC0kFFaA%3D), [Conceptual Caption](https://facevcstandard.blob.core.windows.net/t-zhitang/release/improved_vq_diffusion/cc_learnable.pth?sv=2021-10-04&st=2023-05-30T06%3A33%3A35Z&se=2030-05-31T06%3A33%3A00Z&sr=b&sp=r&sig=8rgV0AZgTyqBNzEPKKJpge%2FZi5vMc0oj7wyn%2BP0BRl0%3D) and [MSCOCO](https://facevcstandard.blob.core.windows.net/t-zhitang/release/improved_vq_diffusion/coco_learnable.pth?sv=2021-10-04&st=2023-05-30T06%3A33%3A54Z&se=2030-05-31T06%3A33%3A00Z&sr=b&sp=r&sig=KHTytyH1ez%2FevQ92oyuamuSiVE6kLguuEqXGRHxapCU%3D) dataset.

We provide the VQVAE models on [FFHQ](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/taming_dvae/vqgan_ffhq_f16_1024.pth?sv=2021-10-04&st=2023-05-30T06%3A45%3A10Z&se=2030-05-31T06%3A45%3A00Z&sr=b&sp=r&sig=LmAGouBRjAVtzZVb%2FLlbUyJU2iS6HZZkGo%2BshBYYYxI%3D), [OpenImages](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/taming_dvae/taming_f8_8192_openimages_last.pth?sv=2021-10-04&st=2023-05-30T06%3A45%3A45Z&se=2030-05-31T06%3A45%3A00Z&sr=b&sp=r&sig=PSQU46nW9ulhMSC7bzgoMrVMxtuw7bDnuBz0c%2FDlS5A%3D), and [ImageNet](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/taming_dvae/vqgan_imagenet_f16_16384.pth?sv=2021-10-04&st=2023-05-30T06%3A46%3A04Z&se=2030-05-31T06%3A46%3A00Z&sr=b&sp=r&sig=hhHJ1uWp6EvnJv%2Fwt%2BeQ%2Fip4h2ae7JOFuyt70GN0HkA%3D) datasets, these models are from [Taming Transformer](https://github.com/CompVis/taming-transformers), we provide them here for convenient. Please put them under OUTPUT/pretrained_model/taming_dvae/ .

To support ITHQ dataset, we trained a new VQVAE model on [ITHQ](https://facevcstandard.blob.core.windows.net/t-zhitang/release/improved_vq_diffusion/ithq_vqvae.pth?sv=2021-10-04&st=2023-05-30T06%3A34%3A08Z&se=2030-05-31T06%3A34%3A00Z&sr=b&sp=r&sig=gLVXl%2Bub9PoCfA8gRWgBTJXO6%2FIhSuIVVQrENQw1ils%3D) dataset.


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
