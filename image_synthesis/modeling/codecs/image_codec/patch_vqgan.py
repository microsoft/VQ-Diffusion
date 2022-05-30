from numpy.core.shape_base import block
from numpy.lib import stride_tricks
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random

from torch.nn.modules.linear import Linear
from image_synthesis.utils.misc import instantiate_from_config
from image_synthesis.modeling.codecs.base_codec import BaseCodec
# from image_synthesis.modeling.modules.vqgan_loss.vqperceptual import VQLPIPSWithDiscriminator
from image_synthesis.modeling.utils.misc import mask_with_top_k, logits_top_k, get_token_type
from image_synthesis.distributed.distributed import all_reduce

# class for quantization
# class for quantization
class EMAVectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta,
                masked_n_e_ratio=0,#1.0/4, 
                embed_init_scale=1.0,
                decay = 0.99,
                embed_ema=True,
                get_embed_type='retrive',
                distance_type='euclidean',
        ):
        super(EMAVectorQuantizer, self).__init__()
        self.n_e = n_e
        self.masked_n_e_ratio = masked_n_e_ratio
        self.e_dim = e_dim
        self.beta = beta
        self.decay = decay 
        self.embed_ema = embed_ema

        self.get_embed_type = get_embed_type
        self.distance_type = distance_type

        if self.embed_ema:
            self.eps = 1.0e-5
            embed = torch.randn(n_e, e_dim)
            # embed = torch.zeros(n_e, e_dim)
            # embed.data.uniform_(-embed_init_scale / self.n_e, embed_init_scale / self.n_e)
            self.register_buffer("embedding", embed)
            self.register_buffer("cluster_size", torch.zeros(n_e))
            self.register_buffer("embedding_avg", embed.clone())
        else:
            self.embedding = nn.Embedding(self.n_e, self.e_dim)
            self.embedding.weight.data.uniform_(-embed_init_scale / self.n_e, embed_init_scale / self.n_e)

        self.masked_embed_start = self.n_e - int(self.masked_n_e_ratio * self.n_e)

        if self.distance_type == 'learned':
            self.distance_fc = nn.Linear(self.e_dim, self.n_e)

    @property
    def norm_feat(self):
        return self.distance_type in ['cosine', 'sinkhorn']
    
    @property
    def embed_weight(self):
        if isinstance(self.embedding, nn.Embedding):
            return self.embedding.weight
        else:
            return self.embedding
    
    def norm_embedding(self):
        if self.training:
            with torch.no_grad():
                w = self.embed_weight.data.clone()
                w = F.normalize(w, dim=1, p=2)
                if isinstance(self.embedding, nn.Embedding):
                    self.embedding.weight.copy_(w)
                else:
                    self.embedding.copy_(w)


    def _quantize(self, z, token_type=None):
        """
            z: L x D
            token_type: L, 1 denote unmasked token, other masked token
        """
        if self.distance_type == 'euclidean':
            d = torch.sum(z ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embed_weight**2, dim=1) - 2 * \
                torch.matmul(z, self.embed_weight.t())
        elif self.distance_type == 'cosine':
            d = 0 - torch.einsum('ld,nd->ln', z, self.embed_weight) # BHW x N
        else:
            raise NotImplementedError('distance not implemented for {}'.format(self.distance_type))

        # find closest encodings 
        # import pdb; pdb.set_trace()
        if token_type is None or self.masked_embed_start == self.n_e:
            min_encoding_indices = torch.argmin(d, dim=1) # L
        else:
            min_encoding_indices = torch.zeros(z.shape[0]).long().to(z.device)
            idx = token_type == 1
            if idx.sum() > 0:
                d_ = d[idx][:, :self.masked_embed_start] # l x n
                indices_ = torch.argmin(d_, dim=1)
                min_encoding_indices[idx] = indices_
            idx = token_type != 1
            if idx.sum() > 0:
                d_ = d[idx][:, self.masked_embed_start:] # l x n
                indices_ = torch.argmin(d_, dim=1) + self.masked_embed_start
                min_encoding_indices[idx] = indices_

        if self.get_embed_type == 'matmul':
            min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z)
            min_encodings.scatter_(1, min_encoding_indices.unsqueeze(1), 1)
            # import pdb; pdb.set_trace()
            z_q = torch.matmul(min_encodings, self.embed_weight)#.view(z.shape)
        elif self.get_embed_type == 'retrive':
            z_q = F.embedding(min_encoding_indices, self.embed_weight)#.view(z.shape)
        else:
            raise NotImplementedError

        return z_q, min_encoding_indices

    def forward(self, z, token_type=None):
        """
            z: B x C x H x W
            token_type: B x 1 x H x W
        """
        if self.distance_type in ['sinkhorn', 'cosine']:
            # need to norm feat and weight embedding    
            self.norm_embedding()            
            z = F.normalize(z, dim=1, p=2)

        # reshape z -> (batch, height, width, channel) and flatten
        batch_size, _, height, width = z.shape
        # import pdb; pdb.set_trace()
        z = z.permute(0, 2, 3, 1).contiguous() # B x H x W x C
        z_flattened = z.view(-1, self.e_dim) # BHW x C
        if token_type is not None:
            token_type_flattened = token_type.view(-1)
        else:
            token_type_flattened = None

        z_q, min_encoding_indices = self._quantize(z_flattened, token_type_flattened)
        z_q = z_q.view(batch_size, height, width, -1) #.permute(0, 2, 3, 1).contiguous()

        if self.training and self.embed_ema:
            # import pdb; pdb.set_trace()
            assert self.distance_type in ['euclidean', 'cosine']
            indices_onehot = F.one_hot(min_encoding_indices, self.n_e).to(z_flattened.dtype) # L x n_e
            indices_onehot_sum = indices_onehot.sum(0) # n_e
            z_sum = (z_flattened.transpose(0, 1) @ indices_onehot).transpose(0, 1) # n_e x D

            all_reduce(indices_onehot_sum)
            all_reduce(z_sum)

            self.cluster_size.data.mul_(self.decay).add_(indices_onehot_sum, alpha=1 - self.decay)
            self.embedding_avg.data.mul_(self.decay).add_(z_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_e * self.eps) * n
            embed_normalized = self.embedding_avg / cluster_size.unsqueeze(1)
            self.embedding.data.copy_(embed_normalized)
            # print((self.embed > 1.0e-20).abs().sum())

        if self.embed_ema:
            loss = (z_q.detach() - z).pow(2).mean()
        else:
            # compute loss for embedding
            loss = torch.mean((z_q.detach()-z).pow(2)) + self.beta * torch.mean((z_q - z.detach()).pow(2))

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        # used_quantize_embed = torch.zeros_like(loss) + min_encoding_indices.unique().shape[0]
        # used_quantize_embed = all_reduce(used_quantize_embed) / get_world_size()

        output = {
            'quantize': z_q,
            'used_quantize_embed': torch.zeros_like(loss) + min_encoding_indices.unique().shape[0],
            'quantize_loss': loss,
            'index': min_encoding_indices.view(batch_size, height, width)
        }
        if token_type_flattened is not None:
            unmasked_num_token = all_reduce((token_type_flattened == 1).sum())
            masked_num_token = all_reduce((token_type_flattened != 1).sum())
            output['unmasked_num_token'] = unmasked_num_token
            output['masked_num_token'] = masked_num_token

        return output


    def only_get_indices(self, z, token_type=None):
        """
            z: B x C x H x W
            token_type: B x 1 x H x W
        """
        if self.distance_type in ['sinkhorn', 'cosine']:
            # need to norm feat and weight embedding    
            self.norm_embedding()            
            z = F.normalize(z, dim=1, p=2)

        # reshape z -> (batch, height, width, channel) and flatten
        batch_size, _, height, width = z.shape
        # import pdb; pdb.set_trace()
        z = z.permute(0, 2, 3, 1).contiguous() # B x H x W x C
        z_flattened = z.view(-1, self.e_dim) # BHW x C
        if token_type is not None:
            token_type_flattened = token_type.view(-1)
        else:
            token_type_flattened = None

        _, min_encoding_indices = self._quantize(z_flattened, token_type_flattened)
        min_encoding_indices = min_encoding_indices.view(batch_size, height, width)

        return min_encoding_indices 

    def get_codebook_entry(self, indices, shape):
        # import pdb; pdb.set_trace()

        # shape specifying (batch, height, width)
        if self.get_embed_type == 'matmul':
            min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
            min_encodings.scatter_(1, indices[:,None], 1)
            # get quantized latent vectors
            z_q = torch.matmul(min_encodings.float(), self.embed_weight)
        elif self.get_embed_type == 'retrive':
            z_q = F.embedding(indices, self.embed_weight)
        else:
            raise NotImplementedError

        if shape is not None:
            z_q = z_q.view(*shape, -1) # B x H x W x C

            if len(z_q.shape) == 4:
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

# blocks for encoder and decoder
def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, upsample_type='interpolate'):
        super().__init__()
        self.upsample_type = upsample_type
        self.with_conv = with_conv
        if self.upsample_type == 'conv':
            self.sample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        if self.upsample_type == 'conv':
            x = self.sample(x)
        else:
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0.0, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), scale_by_2=None, num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=False, **ignore_kwargs):
        super().__init__()
        

        if isinstance(resolution, int):
            resolution = [resolution, resolution] # H, W
        elif isinstance(resolution, (tuple, list)):
            resolution = list(resolution)
        else:
            raise ValueError('Unknown type of resolution:', resolution)
            
        attn_resolutions_ = []
        for ar in attn_resolutions:
            if isinstance(ar, (list, tuple)):
                attn_resolutions_.append(list(ar))
            else:
                attn_resolutions_.append([ar, ar])
        attn_resolutions = attn_resolutions_
        
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn

            if scale_by_2 is None:
                if i_level != self.num_resolutions-1:
                    down.downsample = Downsample(block_in, resamp_with_conv)
                    curr_res = [r // 2 for r in curr_res]
            else:
                if scale_by_2[i_level]:
                    down.downsample = Downsample(block_in, resamp_with_conv)
                    curr_res = [r // 2 for r in curr_res]  
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x):
        #assert x.shape[2] == self.resolution[0] and x.shape[3] == self.resolution[1], "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)

            if getattr(self.down[i_level], 'downsample', None) is not None:
                h = self.down[i_level].downsample(hs[-1])

            if i_level != self.num_resolutions-1:
                # hs.append(self.down[i_level].downsample(hs[-1]))
                hs.append(h)

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), scale_by_2=None, num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 resolution, z_channels, **ignorekwargs):
        super().__init__()
        
        if isinstance(resolution, int):
            resolution = [resolution, resolution] # H, W
        elif isinstance(resolution, (tuple, list)):
            resolution = list(resolution)
        else:
            raise ValueError('Unknown type of resolution:', resolution)
            
        attn_resolutions_ = []
        for ar in attn_resolutions:
            if isinstance(ar, (list, tuple)):
                attn_resolutions_.append(list(ar))
            else:
                attn_resolutions_.append([ar, ar])
        attn_resolutions = attn_resolutions_

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution 
        self.requires_image = False

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        if scale_by_2 is None:
            curr_res = [r // 2**(self.num_resolutions-1) for r in self.resolution]
        else:
            scale_factor = sum([int(s) for s in scale_by_2])
            curr_res = [r // 2**scale_factor for r in self.resolution]

        self.z_shape = (1, z_channels, curr_res[0], curr_res[1])
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if scale_by_2 is None:
                if i_level != 0:
                    up.upsample = Upsample(block_in, resamp_with_conv)
                    curr_res = [r * 2 for r in curr_res]
            else:
                if scale_by_2[i_level]:
                    up.upsample = Upsample(block_in, resamp_with_conv)
                    curr_res = [r * 2 for r in curr_res]
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, **kwargs):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            # if i_level != 0:
            if getattr(self.up[i_level], 'upsample', None) is not None:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class PatchVQGAN(BaseCodec):
    def __init__(self,
                 *,
                 encoder_config,
                 decoder_config,
                 lossconfig=None,
                 n_embed,
                 embed_dim,
                 ignore_keys=[],
                 data_info={'key': 'image'},
                 quantizer_type='VQ',
                 quantizer_dis_type='euclidean',
                 decay = 0.99,
                 trainable=False,
                 ckpt_path=None,
                 token_shape=None
                 ):
        super().__init__()
        self.encoder = instantiate_from_config(encoder_config) # Encoder(**encoder_config)
        self.decoder = instantiate_from_config(decoder_config) # Decoder(**decoder_config)
        if quantizer_type == 'EMAVQ':
            self.quantize = EMAVectorQuantizer(n_embed, embed_dim, beta=0.25, decay = decay, distance_type=quantizer_dis_type)
            print('using EMA vector Quantizer')
        elif quantizer_type == 'PQEMAVQ':
            self.quantize = PQEMAVectorQuantizer(n_embed, embed_dim, beta=0.25,decay = decay, distance_type=quantizer_dis_type)
            print('using PQ EMA vector Quantizer')
        elif quantizer_type == 'VQ':
            self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        else:
            raise NotImplementedError
        # import pdb; pdb.set_trace()
        self.quant_conv = torch.nn.Conv2d(encoder_config['params']["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, decoder_config['params']["z_channels"], 1)

        self.data_info = data_info
    
        if lossconfig is not None and trainable:
            self.loss = instantiate_from_config(lossconfig)
        else:
            self.loss = None
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        self.trainable = trainable
        self._set_trainable()

        self.token_shape = token_shape

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        if 'model' in sd:
            sd = sd['model']
        else:
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("VQGAN: Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"VQGAN: Restored from {path}")

    @property
    def device(self):
        return self.quant_conv.weight.device

    def pre_process(self, data):
        data = data.to(self.device)
        data = data / 127.5 - 1.0
        return data

    def multi_pixels_with_mask(self, data, mask):
        if data.max() > 1:
            raise ValueError('The data need to be preprocessed!')
        mask = mask.to(self.device)
        data = data * mask
        data[~mask.repeat(1,3,1,1)] = -1.0
        return data

    def post_process(self, data):
        data = (data + 1.0) * 127.5
        data = torch.clamp(data, min=0.0, max=255.0)
        return data

    def get_number_of_tokens(self):
        return self.quantize.n_e

    def get_tokens(self, data, mask=None, return_token_index=False, **kwargs):
        data = self.pre_process(data)
        x = self.encoder(data)
        x = self.quant_conv(x)
        idx = self.quantize(x)['index']
        if self.token_shape is None:
            self.token_shape = idx.shape[1:3]

        if self.decoder.requires_image:
            self.mask_im_tmp = self.multi_pixels_with_mask(data, mask)

        output = {}
        output['token'] = idx.view(idx.shape[0], -1)

        # import pdb; pdb.set_trace()
        if mask is not None: # mask should be B x 1 x H x W
            # downsampling
            # mask = F.interpolate(mask.float(), size=idx_mask.shape[-2:]).to(torch.bool)
            token_type = get_token_type(mask, self.token_shape) # B x 1 x H x W
            mask = token_type == 1
            output = {
                'target': idx.view(idx.shape[0], -1).clone(),
                'mask': mask.view(mask.shape[0], -1),
                'token': idx.view(idx.shape[0], -1),
                'token_type': token_type.view(token_type.shape[0], -1),
            }
        else:
            output = {
                'token': idx.view(idx.shape[0], -1)
                }

        # get token index
        # used for computing token frequency
        if return_token_index:
            token_index = output['token'] #.view(-1)
            output['token_index'] = token_index

        return output


    def decode(self, token):
        assert self.token_shape is not None
        # import pdb; pdb.set_trace()
        bhw = (token.shape[0], self.token_shape[0], self.token_shape[1])
        quant = self.quantize.get_codebook_entry(token.view(-1), shape=bhw)
        quant = self.post_quant_conv(quant)
        if self.decoder.requires_image:
            rec = self.decoder(quant, self.mask_im_tmp)
            self.mask_im_tmp = None
        else:
            rec = self.decoder(quant)
        rec = self.post_process(rec)
        return rec


    def get_rec_loss(self, input, rec):
        if input.max() > 1:
            input = self.pre_process(input)
        if rec.max() > 1:
            rec = self.pre_process(rec)

        rec_loss = F.mse_loss(rec, input)
        return rec_loss


    @torch.no_grad()
    def sample(self, batch):

        data = self.pre_process(batch[self.data_info['key']])
        x = self.encoder(data)
        x = self.quant_conv(x)
        quant = self.quantize(x)['quantize']
        quant = self.post_quant_conv(quant)
        if self.decoder.requires_image:
            mask_im = self.multi_pixels_with_mask(data, batch['mask'])
            rec = self.decoder(quant, mask_im)
        else:
            rec = self.decoder(quant)
        rec = self.post_process(rec)

        out = {'input': batch[self.data_info['key']], 'reconstruction': rec}
        if self.decoder.requires_image:
            out['mask_input'] = self.post_process(mask_im)
            out['mask'] = batch['mask'] * 255
            # import pdb; pdb.set_trace()
        return out

    def get_last_layer(self):
        if isinstance(self.decoder, Decoder):
            return self.decoder.conv_out.weight
        elif isinstance(self.decoder, PatchDecoder):
            return self.decoder.post_layer.weight
        elif isinstance(self.decoder, Patch8x8Decoder):
            return self.decoder.post_layer.weight
        else:
            return self.decoder.patch_de_embed.proj.weight

    def parameters(self, recurse=True, name=None):
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            if name == 'generator':
                params = list(self.encoder.parameters())+ \
                         list(self.decoder.parameters())+\
                         list(self.quantize.parameters())+\
                         list(self.quant_conv.parameters())+\
                         list(self.post_quant_conv.parameters())
            elif name == 'discriminator':
                params = self.loss.discriminator.parameters()
            else:
                raise ValueError("Unknown type of name {}".format(name))
            return params

    def forward(self, batch, name='none', return_loss=True, step=0, **kwargs):
        
        if name == 'generator':
            input = self.pre_process(batch[self.data_info['key']])
            x = self.encoder(input)
            x = self.quant_conv(x)
            quant_out = self.quantize(x)
            quant = quant_out['quantize']
            emb_loss = quant_out['quantize_loss']

            # recconstruction
            quant = self.post_quant_conv(quant)
            if self.decoder.requires_image:
                rec = self.decoder(quant, self.multi_pixels_with_mask(input, batch['mask']))
            else:
                rec = self.decoder(quant)
            # save some tensors for 
            self.input_tmp = input 
            self.rec_tmp = rec 

            if isinstance(self.loss, VQLPIPSWithDiscriminator):
                output = self.loss(codebook_loss=emb_loss,
                                inputs=input, 
                                reconstructions=rec, 
                                optimizer_name=name, 
                                global_step=step, 
                                last_layer=self.get_last_layer())
            else:
                raise NotImplementedError('{}'.format(type(self.loss)))

        elif name == 'discriminator':
            if isinstance(self.loss, VQLPIPSWithDiscriminator):
                output = self.loss(codebook_loss=None,
                                inputs=self.input_tmp, 
                                reconstructions=self.rec_tmp, 
                                optimizer_name=name, 
                                global_step=step, 
                                last_layer=self.get_last_layer())
            else:
                raise NotImplementedError('{}'.format(type(self.loss)))
        else:
            raise NotImplementedError('{}'.format(name))
        return output
