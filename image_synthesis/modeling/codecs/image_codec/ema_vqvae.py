import torch
import torch.nn as nn
from omegaconf import OmegaConf
import sys
sys.path.append("..")
# sys.path.append("../image_synthesis")
import os
import torchvision.transforms.functional as TF
import PIL
from image_synthesis.modeling.codecs.base_codec import BaseCodec
from einops import rearrange
import math
import yaml
from image_synthesis.utils.misc import instantiate_from_config

class Encoder(nn.Module):
    def __init__(self, encoder, quant_conv, quantize):
        super().__init__()
        self.encoder = encoder
        self.quant_conv = quant_conv
        self.quantize = quantize

    @torch.no_grad()
    def forward(self, x):
        x = 2*x - 1
        h = self.encoder(x)
        h = self.quant_conv(h)
        # quant, _, [_, _, indices] = self.quantize(h)
        # return indices.view(x.shape[0], -1)
        indices = self.quantize.only_get_indices(h)
        return indices.view(x.shape[0], -1)

class Decoder(nn.Module):
    def __init__(self, decoder, post_quant_conv, quantize, w=16, h=16):
        super().__init__()
        self.decoder = decoder
        self.post_quant_conv = post_quant_conv
        self.quantize = quantize
        self.w = w
        self.h = h

    @torch.no_grad()
    def forward(self, indices):
        z = self.quantize.get_codebook_entry(indices.view(-1), shape=(indices.shape[0], self.h, self.w))
        quant = self.post_quant_conv(z)
        dec = self.decoder(quant)
        x = torch.clamp(dec, -1., 1.)
        x = (x + 1.)/2.
        return x

class PatchVQVAE(BaseCodec):
    def __init__(
            self, 
            trainable=False,
            token_shape=[16,16],
        ):
        super().__init__()
        
        config_path = "OUTPUT/pretrained_model/taming_dvae/config.yaml"
        ckpt_path="OUTPUT/pretrained_model/taming_dvae/ithq_vqvae.pth"
        model = self.LoadModel(config_path, ckpt_path)

        self.enc = Encoder(model.encoder, model.quant_conv, model.quantize)
        self.dec = Decoder(model.decoder, model.post_quant_conv, model.quantize, token_shape[0], token_shape[1])

        self.num_tokens = 4096
    
        self.trainable = trainable
        self.token_shape = token_shape
        self._set_trainable()

    def LoadModel(self, config_path, ckpt_path):
        with open(config_path) as f:
            config = yaml.full_load(f)
        model = instantiate_from_config(config['model'])
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        model.load_state_dict(sd, strict=False)
        return model


    def half(self):             # not sure if it's right
        """
        overwrite this function
        """
        from dall_e.utils import Conv2d
        for n, m in self.named_modules():
            if isinstance(m, Conv2d) and m.use_float16:
                print(n)
                m._apply(lambda t: t.half() if t.is_floating_point() else t)

        return self

    @property
    def device(self):
        # import pdb; pdb.set_trace()
        return self.enc.quant_conv.weight.device

    def preprocess(self, imgs):
        """
        imgs: B x C x H x W, in the range 0-255
        """
        imgs = imgs.div(255) # map to 0 - 1
        return imgs
        # return map_pixels(imgs)   
    
    def postprocess(self, imgs):
        """
        imgs: B x C x H x W, in the range 0-1
        """
        imgs = imgs * 255
        return imgs

    def get_tokens(self, imgs, **kwargs):
        imgs = self.preprocess(imgs)
        code = self.enc(imgs)
        output = {'token': code}
        # output = {'token': rearrange(code, 'b h w -> b (h w)')}
        return output

    def decode(self, img_seq):
        b, n = img_seq.shape
        # if self.token_shape is not None:
        #     img_seq = img_seq.view(b, self.token_shape[0], self.token_shape[1])
        # else:
        #     img_seq = rearrange(img_seq, 'b (h w) -> b h w', h = int(sqrt(n)))
        img_seq = rearrange(img_seq, 'b (h w) -> b h w', h = int(math.sqrt(n)))

        x_rec = self.dec(img_seq)
        x_rec = self.postprocess(x_rec)
        return x_rec


