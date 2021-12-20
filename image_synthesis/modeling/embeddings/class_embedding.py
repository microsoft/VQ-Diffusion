import torch
import torch.nn as nn
from .base_embedding import BaseEmbedding

class ClassEmbedding(BaseEmbedding):
    def __init__(self, 
                 num_embed=1000,
                 embed_dim=512,
                 identity=False,
                 trainable=True,
        ):
        super().__init__()
        self.identity = identity
        self.trainable = trainable
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        if self.identity == False:
        
            self.emb = nn.Embedding(self.num_embed, embed_dim)
            self._set_trainable()

    def forward(self, index, **kwargs):
        """
        index: B x L, index
        mask: B x L, bool type. The value of False indicating padded index
        """
        if self.identity == True:
            return index
        else:
            emb = self.emb(index).unsqueeze(1)
            return emb

