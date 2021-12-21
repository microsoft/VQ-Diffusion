# Vector Quantized Diffusion Model for Text-to-Image Synthesis

## Overview

This is the official repo for the paper: [Vector Quantized Diffusion Model for Text-to-Image Synthesis](https://arxiv.org/pdf/2111.14822.pdf).

> VQ-Diffusion is based on a VQ-VAE whose latent space is modeled by a conditional variant of the recently developed Denoising Diffusion Probabilistic Model (DDPM). It produces significantly better text-to-image generation results when compared with Autoregressive models with similar numbers of parameters. Compared with previous GAN-based methods, VQ-Diffusion can handle more complex scenes and improve the synthesized image quality by a large margin.

## Framework

<img src='figures/framework.png' width='600'>

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
We release four text-to-image pretrained model, trained on [Conceptual Caption](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/CC_pretrained.pth?sv=2019-12-12&st=2021-12-18T05%3A39%3A41Z&se=2028-12-19T05%3A39%3A00Z&sr=b&sp=r&sig=o6mT3jVHjJvmmwmWqt3jWv35tBlM%2BpMjf3XmlAS7osI%3D), [MSCOCO](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/coco_pretrained.pth?sv=2019-12-12&st=2021-12-18T05%3A40%3A15Z&se=2028-12-19T05%3A40%3A00Z&sr=b&sp=r&sig=RCzwxUEw21xagqqAvpM8dIyO4ZK1s0Pkt3uc7R1WHx4%3D), [CUB200](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/cub_pretrained.pth?sv=2019-12-12&st=2021-12-18T05%3A40%3A44Z&se=2028-12-19T05%3A40%3A00Z&sr=b&sp=r&sig=0BuaBa1Fro6jvnHQ5zAYiY0ecgu7PEMs%2BK7EtqmjL1M%3D), and [LAION-human](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/human_pretrained.pth?sv=2019-12-12&st=2021-12-18T05%3A41%3A14Z&se=2028-12-19T05%3A41%3A00Z&sr=b&sp=r&sig=tK%2BCuX9jpikRmGKg4lvx2MBOvU2JRwu7sq9pKNAenh4%3D) datasets. Also, we release the [ImageNet](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/imagenet_pretrained.pth?sv=2019-12-12&st=2021-12-18T05%3A42%3A24Z&se=2028-12-19T05%3A42%3A00Z&sr=b&sp=r&sig=TOS5fddB5vSGKTLBsPMC5LhpyWRbZTlQKeIJrlQmiyg%3D) pretrained model, and provide the [CLIP](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/ViT-B-32.pt?sv=2019-12-12&st=2021-12-18T05%3A38%3A59Z&se=2028-03-19T05%3A38%3A00Z&sr=b&sp=r&sig=vjKexxaBROiGlS1iUkU7MzC0pMFjJBqSffj0SZCtVw0%3D) pretrained model for convenient. These should be put under OUTPUT/pretrained_model/ .
These pretrained model file may be large because they are training checkpoints, which contains gradient information, optimizer information, ema model and others.

Besides, we provide the VQVAE models on [FFHQ](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/taming_dvae/vqgan_ffhq_f16_1024.pth?sv=2019-12-12&st=2021-12-21T04%3A27%3A57Z&se=2028-12-22T04%3A27%3A00Z&sr=b&sp=r&sig=zmHBBWw81qMgtYg8fFU1ivFp0iBRky17hHxiRYy%2BaJ8%3D), [OpenImages](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/taming_dvae/taming_f8_8192_openimages_last.pth?sv=2019-12-12&st=2021-12-21T04%3A27%3A16Z&se=2028-12-22T04%3A27%3A00Z&sr=b&sp=r&sig=1emEQtO9hxA2L1%2FnLRf5NsV327keGAZKBddNabG3whM%3D), and [imagenet](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/taming_dvae/vqgan_imagenet_f16_16384.pth?sv=2019-12-12&st=2021-12-21T04%3A28%3A27Z&se=2028-12-22T04%3A28%3A00Z&sr=b&sp=r&sig=W5LJJ23tIdkLS7QIJJcxdTdo164i3X%2BCgdDZaAREapE%3D) datasets, these model are from [Taming Transformer](https://github.com/CompVis/taming-transformers), we provide them here for convenient. Please put them under OUTPUT/pretrained_model/taming_dvae/ .

## Inference
To generate image from given text:
```
from inference_VQ_Diffusion import VQ_Diffusion
VQ_Diffusion_model = VQ_Diffusion(config='OUTPUT/pretrained_model/config_text.yaml', path='OUTPUT/pretrained_model/human_pretrained.pth')
VQ_Diffusion_model.inference_generate_sample_with_condition("a beautiful smiling woman",truncation_rate=0.85, save_root="RESULT",batch_size=4)
VQ_Diffusion_model.inference_generate_sample_with_condition("a woman in yellow dress",truncation_rate=0.85, save_root="RESULT",batch_size=4,fast=2) # for fast inference
```
You may change human_pretrained.pth to other pretrained model to test different text.

To generate image from given ImageNet class label:
```
from inference_VQ_Diffusion import VQ_Diffusion
VQ_Diffusion_model = VQ_Diffusion(config='OUTPUT/pretrained_model/config_imagenet.yaml', path='OUTPUT/pretrained_model/imagenet_pretrained.pth')
VQ_Diffusion_model.inference_generate_sample_with_class(407,truncation_rate=0.86, save_root="RESULT",batch_size=4)
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
