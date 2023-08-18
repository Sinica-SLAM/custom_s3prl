
import torch
from torch import nn
from torch import Tensor
from torch.nn import MultiheadAttention
from s3prl.upstream.featurizer import tolist
from typing import List

class BaseFusionModule(nn.Module):
    def __init__(self, featurizer1, featurizer2, **kwargs):
        super().__init__()
        self.name = self.__class__.__name__
        assert featurizer1.output_dim == featurizer2.output_dim
        self.upstream_dim = featurizer1.output_dim

        assert featurizer1.downsample_rate == featurizer2.downsample_rate
        self.downsample_rate = featurizer1.downsample_rate

        if self.__class__.__name__ == 'BaseFusionModule':
            self.showinfo()

    def forward(self, features1: Tensor, features2: Tensor, f_lens: List[int]):
        raise NotImplementedError
    
    def showinfo(self):
        print(f"[Fusioner] - Name: {self.name}")
        print(f"[Fusioner] - Upstream dim: {self.upstream_dim}")
        print(f"[Fusioner] - Downsample rate: {self.downsample_rate}")


class lambda_sum(BaseFusionModule):
    def __init__(self, featurizer1, featurizer2, init_lamb=0.0, **kwargs):
        super().__init__(featurizer1, featurizer2, **kwargs)
        self.lamb = nn.Parameter(torch.tensor(init_lamb))

        self.showinfo()

    def forward(self, features1: Tensor, features2: Tensor, f_lens: List[int]):
        features1 = tolist(f_lens, features1)
        features2 = tolist(f_lens, features2)
        self.l = torch.sigmoid(self.lamb)
        return [self.l * f1 + (1 - self.l) * f2 for f1, f2 in zip(features1, features2)]


class cat_in_dim(BaseFusionModule):
    def __init__(self, featurizer1, featurizer2, **kwargs):
        super().__init__(featurizer1, featurizer2, **kwargs)
        self.upstream_dim = 2 * self.upstream_dim

        self.showinfo()

    def forward(self, features1: Tensor, features2: Tensor, f_lens: List[int]):
        features1 = tolist(f_lens, features1)
        features2 = tolist(f_lens, features2)
        return [torch.cat([f1, f2], dim=-1) for f1, f2 in zip(features1, features2)]


class cat_in_time(BaseFusionModule):
    def __init__(self, featurizer1, featurizer2, **kwargs):
        super().__init__(featurizer1, featurizer2, **kwargs)
        assert self.downsample_rate % 2 == 0
        self.downsample_rate = self.downsample_rate // 2

        self.showinfo()

    def forward(self, features1: Tensor, features2: Tensor, f_lens: List[int]): # (B, T, D)
        features1 = tolist(f_lens, features1)
        features2 = tolist(f_lens, features2)
        return [torch.cat([f1, f2], dim=-2) for f1, f2 in zip(features1, features2)] # B * (2T, D)


class cross_attention(BaseFusionModule):
    def __init__(self, featurizer1, featurizer2, fusion_heads=8, **kwargs):
        super().__init__(featurizer1, featurizer2, **kwargs)
        self.attention = MultiheadAttention(self.upstream_dim, fusion_heads, dropout=0.1, batch_first=True)

        self.showinfo()

    def forward(self, features1: Tensor, features2: Tensor, f_lens: List[int]): # (B, T, D)
        masks = torch.zeros(features1.shape[:-1], dtype=torch.bool, device=features1.device) # (B, T)
        for mask, f_len in zip(masks, f_lens): # (T)
            mask[f_len:] = True
        features, _ = self.attention(features1, features2, features2, key_padding_mask=masks, need_weights=False) # (B, T, D)
        return tolist(f_lens, features + features2) # B * (T, D)


class cross_in_time(BaseFusionModule):
    def __init__(self, featurizer1, featurizer2, **kwargs):
        super().__init__(featurizer1, featurizer2, **kwargs)
        assert self.downsample_rate % 2 == 0
        self.downsample_rate = self.downsample_rate // 2

        self.showinfo()

    def forward(self, features1, features2, f_lens):
        features1 = tolist(f_lens, features1)
        features2 = tolist(f_lens, features2)
        def cross(f1, f2):
            *other, T, D = f1.shape
            return torch.cat([f1, f2], dim=-1).reshape(*other, 2*T, D) 
        return [cross(f1, f2) for f1, f2 in zip(features1, features2)]