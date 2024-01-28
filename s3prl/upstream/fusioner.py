
import torch
from torch import nn
from torch import Tensor
from torch.nn import MultiheadAttention
from s3prl.upstream.featurizer import tolist, padding
from typing import List

class BaseFusionModule(nn.Module):
    def __init__(self, featurizer1, featurizer2):
        super().__init__()
        assert featurizer1.output_dim == featurizer2.output_dim
        self.upstream_dim = featurizer1.output_dim

        self.trainable = False

        assert featurizer1.downsample_rate == featurizer2.downsample_rate
        self.downsample_rate = featurizer1.downsample_rate

        if self.__class__.__name__ == 'BaseFusionModule':
            self.showinfo()

    def forward(self, features1: List[Tensor], features2: List[Tensor]):
        raise NotImplementedError

    def showinfo(self):
        print(f"[Fusioner] - Name: {self.__class__.__name__}")
        print(f"[Fusioner] - Upstream dim: {self.upstream_dim}")
        print(f"[Fusioner] - Downsample rate: {self.downsample_rate}")


class lambda_sum(BaseFusionModule):
    def __init__(self, featurizer1, featurizer2, init_lamb=0.0, fix_lamb=False, **kwargs):
        super().__init__(featurizer1, featurizer2, **kwargs)
        self.fix_lamb = fix_lamb
        if fix_lamb:
            self.l = torch.sigmoid(torch.tensor(init_lamb)).item()
            self.trainable = False
        else:
            self.lamb = nn.Parameter(torch.tensor(init_lamb))
            self.trainable = True

        self.showinfo()

    def showinfo(self):
        super().showinfo()
        if self.fix_lamb:
            print(f"[Fusioner] - Fix Weight: {self.l}")
        else:
            print(f"[Fusioner] - Init Weight: {torch.sigmoid(self.lamb).item()}")

    def forward(self, features1: List[Tensor], features2: List[Tensor]):
        if self.fix_lamb:
            return [self.l * f1 + (1 - self.l) * f2 for f1, f2 in zip(features1, features2)]
        else:
            l = torch.sigmoid(self.lamb)
            return [l * f1 + (1 - l) * f2 for f1, f2 in zip(features1, features2)]


class cat_in_dim(BaseFusionModule):
    def __init__(self, featurizer1, featurizer2, **kwargs):
        super().__init__(featurizer1, featurizer2, **kwargs)
        self.upstream_dim = 2 * self.upstream_dim

        self.trainable = False

        self.showinfo()

    def forward(self, features1: List[Tensor], features2: List[Tensor]):
        return [torch.cat([f1, f2], dim=-1) for f1, f2 in zip(features1, features2)]


class cat_in_time(BaseFusionModule):
    def __init__(self, featurizer1, featurizer2, **kwargs):
        super().__init__(featurizer1, featurizer2, **kwargs)
        assert self.downsample_rate % 2 == 0
        self.downsample_rate = self.downsample_rate // 2

        self.trainable = False

        self.showinfo()

    def forward(self, features1: List[Tensor], features2: List[Tensor]):
        return [torch.cat([f1, f2], dim=-2) for f1, f2 in zip(features1, features2)] # B * (2T, D)


class cross_attention(BaseFusionModule):
    def __init__(self, featurizer1, featurizer2, fusion_heads=8, **kwargs):
        super().__init__(featurizer1, featurizer2, **kwargs)
        self.attention = MultiheadAttention(self.upstream_dim, fusion_heads, dropout=0.1, batch_first=True)
        self.norm_layer = nn.LayerNorm(self.upstream_dim)

        self.trainable = True

        self.showinfo()

    def forward(self, features1: List[Tensor], features2: List[Tensor]):
        # padding
        f_lens1, features1 = padding(features1) # (B, T, D)
        f_lens2, features2 = padding(features2) # (B, T, D)
        assert all([f_len1 == f_len2 for f_len1, f_len2 in zip(f_lens1, f_lens2)])
        f_lens = f_lens1

        # cross attention
        masks = torch.zeros(features1.shape[:-1], dtype=torch.bool, device=features1.device) # (B, T)
        for mask, f_len in zip(masks, f_lens): # (T)
            mask[f_len:] = True
        features, _ = self.attention(features2, features1, features1, key_padding_mask=masks, need_weights=False) # (B, T, D)

        # add & norm
        features = self.norm_layer(features + features1)

        return tolist(f_lens, features) # B * (T, D)


class cross_in_time(BaseFusionModule):
    def __init__(self, featurizer1, featurizer2, **kwargs):
        super().__init__(featurizer1, featurizer2, **kwargs)
        assert self.downsample_rate % 2 == 0
        self.downsample_rate = self.downsample_rate // 2

        self.trainable = False

        self.showinfo()

    def forward(self, features1: List[Tensor], features2: List[Tensor]):
        def cross(f1, f2):
            *other, T, D = f1.shape
            return torch.cat([f1, f2], dim=-1).reshape(*other, 2*T, D)
        return [cross(f1, f2) for f1, f2 in zip(features1, features2)]


class gate_dim(BaseFusionModule):
    def __init__(self, featurizer1, featurizer2, init_value=1.5, **kwargs):
        super().__init__(featurizer1, featurizer2, **kwargs)

        self.trainable = True

        self.gate_values = nn.Parameter(torch.empty(self.upstream_dim).fill_(init_value))

        self.showinfo()

    def _get_gate(self):
        return torch.sigmoid(self.gate_values)

    def forward(self, features1: List[Tensor], features2: List[Tensor]):
        gates = self._get_gate() # gate for each dimension
        # element-wise multiplication of gate and feature
        return [f1 * gate + f2 * (1 - gate) for f1, f2, gate in zip(features1, features2, gates)]