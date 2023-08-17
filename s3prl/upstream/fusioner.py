
import torch
from torch import nn
from torch import Tensor
from typing import List

class BaseFusionModule(nn.Module):
    def __init__(self, upstream_dim, downsample_rate, **kwargs):
        super().__init__()
        self.upstream_dim = upstream_dim
        self.downsample_rate = downsample_rate

    def forward(self, features1: List[Tensor], features2: List[Tensor]):
        raise NotImplementedError


class lambda_sum(BaseFusionModule):
    def __init__(self, upstream_dim, downsample_rate, init_lamb=0.0, **kwargs):
        super().__init__(upstream_dim, downsample_rate, **kwargs)
        self.lamb = nn.Parameter(torch.tensor(init_lamb))

    def forward(self, features1, features2):
        self.l = torch.sigmoid(self.lamb)
        return [self.l * f1 + (1 - self.l) * f2 for f1, f2 in zip(features1, features2)]


class cat_in_dim(BaseFusionModule):
    def __init__(self, upstream_dim, downsample_rate, **kwargs):
        super().__init__(upstream_dim, downsample_rate, **kwargs)
        self.upstream_dim = 2 * upstream_dim

    def forward(self, features1, features2):
        return [torch.cat([f1, f2], dim=-1) for f1, f2 in zip(features1, features2)]


class cat_in_time(BaseFusionModule):
    def __init__(self, upstream_dim, downsample_rate, **kwargs):
        super().__init__(upstream_dim, downsample_rate, **kwargs)
        assert downsample_rate % 2 == 0
        self.downsample_rate = downsample_rate // 2

    def forward(self, features1, features2):
        return [torch.cat([f1, f2], dim=-2) for f1, f2 in zip(features1, features2)]


class cross_in_time(BaseFusionModule):
    def __init__(self, upstream_dim, downsample_rate, **kwargs):
        super().__init__(upstream_dim, downsample_rate, **kwargs)
        assert downsample_rate % 2 == 0
        self.downsample_rate = downsample_rate // 2

    def forward(self, features1, features2):
        def cross(f1, f2):
            *other, T, D = f1.shape
            return torch.cat([f1, f2], dim=-1).reshape(*other, 2*T, D)
        features = [cross(f1, f2) for f1, f2 in zip(features1, features2)]
        return features