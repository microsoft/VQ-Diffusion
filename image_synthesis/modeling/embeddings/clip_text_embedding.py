import torch
import torch.nn as nn
from image_synthesis.modeling.modules.clip import clip
from image_synthesis.modeling.modules.clip import model as clip_model
from .base_embedding import BaseEmbedding

class CLIPTextEmbedding(BaseEmbedding):
    def __init__(self, 
                 clip_name='ViT-B/32',
                 num_embed=49408,
                 normalize=True,
                 pick_last_embedding=True,
                 keep_seq_len_dim=False,
                 additional_last_embedding=False,
                 embed_dim=1024,
        ):
        super().__init__()
        self.num_embed = num_embed
        self.clip_name = clip_name
        self.normalize = normalize
        self.pick_last_embedding = pick_last_embedding
        self.keep_seq_len_dim = keep_seq_len_dim
        self.additional_last_embedding = additional_last_embedding

        model, _ = clip.load(clip_name, device='cpu',jit=False)
        model = clip_model.build_model(model.state_dict())

        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        
        if embed_dim == 1024:
            self.embed_dim = self.text_projection.shape[1]*2    # to fit 1024 dimension of image embedding
        else:
            self.embed_dim = self.text_projection.shape[1]    # original output, 512 dim

        self.trainable = False
        self._set_trainable()

    @property
    def dtype(self):
        return self.transformer.resblocks[0].attn.in_proj_weight.dtype

    def encode_text(self, text):
        text[text < 0] = 0 # some padded text token maybe negative, so set them to 0
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        if self.pick_last_embedding:
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection # [batch_size, transformer.width]
            if self.keep_seq_len_dim:
                x = x.unsqueeze(dim=1) # [batch_size, 1, transformer.width]
        return x



    def forward(self, index, **kwargs):
        """
        index: B x L, index
        mask: B x L, bool type. The value of False indicating padded index
        """
        assert index.dim() == 2 # B x L
        text_feature = self.encode_text(index)

        if self.embed_dim == 1024:
            text_features = torch.cat((text_feature, text_feature), dim=2)
        else:
            text_features = text_feature
        if self.normalize:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if self.additional_last_embedding == True:
            last_feature = text_feature[torch.arange(text_feature.shape[0]), index.argmax(dim=-1)] @ self.text_projection
            if self.keep_seq_len_dim:
                last_feature = last_feature.unsqueeze(dim=1)
            return text_features, last_feature


        return text_features
