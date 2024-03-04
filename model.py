import torch as th
from utils import visualize_patches
import torch.nn.functional as F
import math
from stn import SpatialTransformer


class SA(th.nn.Module):

    def __init__(self, hidden_dim, d_h, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.d_h = d_h

        self.u_q = th.nn.Parameter(th.randn(1, self.hidden_dim, self.d_h))
        th.nn.init.kaiming_uniform_(self.u_q, a=math.sqrt(5))
        self.u_k = th.nn.Parameter(th.randn(1, self.hidden_dim, self.d_h))
        th.nn.init.kaiming_uniform_(self.u_k, a=math.sqrt(5))
        self.u_v = th.nn.Parameter(th.randn(1, self.hidden_dim, self.d_h))
        th.nn.init.kaiming_uniform_(self.u_v, a=math.sqrt(5))

    def forward(self, x):
        cast_u_q = th.broadcast_to(self.u_q, (x.shape[0], self.hidden_dim, self.d_h))
        cast_u_k = th.broadcast_to(self.u_k, (x.shape[0], self.hidden_dim, self.d_h))
        cast_u_v = th.broadcast_to(self.u_v, (x.shape[0], self.hidden_dim, self.d_h))

        query = th.bmm(x, cast_u_q)
        key = th.bmm(x, cast_u_k)
        value = th.bmm(x, cast_u_v)

        a = th.bmm(query, th.transpose(key, 1, 2))
        a = a / self.d_h ** 0.5
        a = F.softmax(a, dim=-1)

        a = th.bmm(a, value)
        return a


class MSA(th.nn.Module):

    def __init__(self, hidden_dim, num_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.d_h = hidden_dim // num_heads

        self.self_attentions = th.nn.ModuleList([SA(self.hidden_dim, self.d_h) for _ in range(self.num_heads)])

        self.u_msa = th.nn.Parameter(th.randn(1, self.num_heads * self.d_h, self.hidden_dim))
        th.nn.init.kaiming_uniform_(self.u_msa, a=math.sqrt(5))

    def forward(self, x):
        cast_u_msa = th.broadcast_to(self.u_msa, (x.shape[0], self.u_msa.shape[1], self.u_msa.shape[2]))

        msa_output = th.concat([sa(x) for sa in self.self_attentions], dim=-1)
        msa_output = th.bmm(msa_output, cast_u_msa)
        return msa_output


class TransformerEncoder(th.nn.Module):

    def __init__(self, num_embeddings, hidden_dim, num_heads, mlp_size, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_embeddings = num_embeddings
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mlp_size = mlp_size
        self.dropout = dropout

        self.ln1 = th.nn.LayerNorm([self.hidden_dim])
        self.ln2 = th.nn.LayerNorm([self.hidden_dim])

        self.msa = MSA(self.hidden_dim, self.num_heads)

        self.mlp = th.nn.Sequential(th.nn.Linear(self.hidden_dim, self.mlp_size),
                                    th.nn.GELU(),
                                    th.nn.Dropout(self.dropout),
                                    th.nn.Linear(self.mlp_size, self.hidden_dim),
                                    th.nn.GELU(),
                                    th.nn.Dropout(self.dropout))

        self.stn = SpatialTransformer(num_embeddings, hidden_dim)

    def forward(self, x):
        out = self.ln1(x)
        out = self.msa(out)

        res_out = out + x

        out = self.ln2(res_out)
        out = self.mlp(out)

        out = out + res_out

        out = self.stn(out)

        return out


class ViT(th.nn.Module):

    def __init__(self,
                 image_size,
                 num_classes,
                 patch_size=16,
                 num_layers=12,
                 hidden_dim=768,
                 num_heads=12,
                 mlp_size=3072,
                 dropout=0.1,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.image_size = image_size
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mlp_size = mlp_size
        self.dropout = dropout
        self.num_channels = 3

        n = self.image_size * self.image_size // (self.patch_size * self.patch_size)
        self.pos_embedding = th.nn.Parameter(th.randn(1, n + 1, self.hidden_dim))

        self.projection = th.nn.Linear(self.patch_size * self.patch_size * self.num_channels, self.hidden_dim)
        self.cls_token = th.nn.Parameter(th.randn(1, 1, self.hidden_dim))

        self.encoders = th.nn.ModuleList([TransformerEncoder(n + 1, self.hidden_dim, self.num_heads, self.mlp_size, self.dropout)
                                          for _ in range(self.num_layers)])

        # MLP width: the width of the classification hidden layer: 3072, tanh activation
        self.cls_head = th.nn.Sequential(th.nn.Linear(self.hidden_dim, self.hidden_dim),
                                         th.nn.Tanh(),
                                         th.nn.Dropout(self.dropout),
                                         th.nn.Linear(self.hidden_dim, self.num_classes),
                                         th.nn.Dropout(self.dropout))

    def forward(self, x):  # x: B x C x H x W
        # split into patches
        x = x.unfold(-2, self.patch_size, self.patch_size)
        x = x.unfold(-2, self.patch_size, self.patch_size)  # x: B x C x H/P x W/P x P x P
        x = x.permute(0, 2, 3, 4, 5, 1)  # x: B x H/P x W/P x P x P x C
        x = x.reshape(x.shape[0],
                      x.shape[1] * x.shape[2],
                      self.patch_size * self.patch_size * self.num_channels)  # x: B x N x P^2*C

        x = self.projection(x)
        token = th.broadcast_to(self.cls_token, (x.shape[0], 1, self.hidden_dim))
        x = th.cat([token, x], dim=1)

        pos_embed = th.broadcast_to(self.pos_embedding,
                                    (x.shape[0], self.pos_embedding.shape[1], self.pos_embedding.shape[2]))
        x = x + pos_embed
        x = F.dropout(x, self.dropout)

        for encoder in self.encoders:
            x = encoder(x)

        x = self.cls_head(x[:, 0, :])
        return x

