import torch
from torch import nn


class BaseCodec(nn.Module):
    
    def get_tokens(self, x, **kwargs):
        """
        Input: 
            x: input data
        Return:
            indices: B x L, the codebook indices, where L is the length 
                    of flattened feature map size
        """
        raise NotImplementedError

    def get_number_of_tokens(self):
        """
        Return: int, the number of tokens
        """
        raise NotImplementedError

    def encode(self, img):
        raise NotImplementedError

    def decode(self, img_seq):
        raise NotImplementedError

    def forward(self, **kwargs):
        raise NotImplementedError

    def train(self, mode=True):
        self.training = mode
        if self.trainable and mode:
            return super().train(True)
        else:
            return super().train(False)

    def _set_trainable(self):
        if not self.trainable:
            for pn, p in self.named_parameters():
                p.requires_grad = False
            self.eval()