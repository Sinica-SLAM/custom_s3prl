import sys
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import MultiheadAttention, Linear

from s3prl.utility.helper import show

from .interfaces import UpstreamBase, SAMPLE_RATE, TOLERABLE_SEQLEN_DIFF

"""use self-attention of all layers of upstream SLL model to get the feature"""

class AttentionPooling(nn.Module):
    def __init__(self, input_dim, select_id):
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = input_dim // 2
        self.select_id = select_id
        self.Q = Linear(input_dim, self.proj_dim)
        self.K = Linear(input_dim, self.proj_dim)

    def forward(self, x): # (..., seq_len, input_dim)
        select_layer = x[..., self.select_id, :] # (..., input_dim)
        quiry_x = self.Q(x)                      # (..., seq_len, proj_dim)
        key_x = self.K(select_layer)             # (..., proj_dim)
        # compute attention weights by dot product of quiry and key
        attn_weights = torch.matmul(quiry_x, key_x.unsqueeze(-1)).squeeze(-1) # (..., seq_len)
        attn_weights = F.softmax(attn_weights / np.sqrt(self.proj_dim), dim=-1) # (..., seq_len)
        # weighted sum of all layers
        attn_x = torch.matmul(attn_weights.unsqueeze(-2), x).squeeze(-2) # (..., input_dim)
        return attn_x

class AttentionPooling2(nn.Module):
    def __init__(self, input_dim, _):
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = input_dim // 2
        self.proj = Linear(input_dim, self.proj_dim)

        self.attn = nn.Parameter(torch.randn(self.proj_dim,1)) # (proj_dim, 1)

    def forward(self, x):     # (..., seq_len, input_dim)
        proj_x = self.proj(x) # (..., seq_len, proj_dim)
        weights = torch.matmul(proj_x, self.attn).squeeze(-1) # (..., seq_len)
        norm_weights = F.softmax(weights / np.sqrt(self.proj_dim), dim=-1) # (..., seq_len)
        attn_x = torch.matmul(norm_weights.unsqueeze(-2), x).squeeze(-2) # (..., input_dim)
        return attn_x

class AttentionPooling3(nn.Module):
    """Multihead version of AttentionPooling2"""
    def __init__(self, idim, num_heads = 4):
        super().__init__()
        assert idim % num_heads == 0, "proj_dim should be divisible by num_heads"
        vdim = idim // num_heads
        kdim = vdim // 2
        self.idim = idim
        self.kdim = kdim
        self.vdim = vdim
        self.num_heads = num_heads

        self.Q   = nn.Parameter(torch.randn(num_heads, vdim, kdim)) # (num_heads, kdim, kdim)
        self.key = nn.Parameter(torch.randn(num_heads, kdim,    1)) # (num_heads, kdim,    1)

    def forward(self, x : Tensor): # (..., seq_len, idim)
        split_x = x.reshape(*x.shape[:-1], self.num_heads, self.vdim) # (..., seq_len, num_heads, vdim)
        split_x = split_x.transpose(-2, -3) # (..., num_heads, seq_len, vdim)
        quiry_x = torch.matmul(split_x, self.Q) # (..., num_heads, seq_len, kdim)
        # compute attention weights by dot product of quiry and key
        attn_weights = torch.matmul(quiry_x, self.key).squeeze(-1) # (..., num_heads, seq_len)
        attn_weights = F.softmax(attn_weights / np.sqrt(self.kdim), dim=-1) # (..., num_heads, seq_len)
        # weighted sum of all layers
        split_y = torch.matmul(attn_weights.unsqueeze(-2), split_x).squeeze(-2) # (..., num_heads, vdim)
        y = split_y.reshape(*x.shape[:-2], self.idim) # (..., idim)
        return y


class TFFeaturizer(nn.Module):
    def __init__(
        self,
        upstream: UpstreamBase,
        upstream_device: str = "cuda",
        layer_selection: int = None,
        normalize: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.name = "TFFeaturizer"
        print("[TFFeaturizer] init")
        assert layer_selection is not None, "layer_selection should be specified"
        self.layer_selection = layer_selection
        self.normalize = normalize
        self.downsample_rate = upstream.get_downsample_rates("hidden_states")
        self.upstream_device = upstream_device
        self.output_dim = self.get_output_dim(upstream, upstream_device)
        self.atten_pooling = AttentionPooling2(self.output_dim, self.layer_selection)

    def get_output_dim(self, upstream: UpstreamBase, upstream_device: str):
        upstream.eval()
        paired_wavs = [torch.randn(SAMPLE_RATE).to(upstream_device)]
        with torch.no_grad():
            paired_features = upstream(paired_wavs)
        feature = self._select_feature(paired_features)
        if isinstance(feature, (list, tuple)):
            feature = feature[0]
        output_dim = feature.size(-1)
        return output_dim

    def _select_feature(self, features):
        feature = features.get("hidden_states")

        if isinstance(feature, dict):
            feature = list(feature.values())

        return feature

    def _attn_sum(self, features: List[Tensor]):
        """use attention pool the calculate the feature"""
        features = torch.stack(features, dim=0)          # L*(N, T, D) -> (L, N, T, D)
        if self.normalize:
            features = F.layer_norm(features, features.shape[3:]) # (L, N, T, D)
        prmt_features = features.permute(2, 1, 0, 3)     # (L, N, T, D) -> (T, N, L, D)
        attn_feature = self.atten_pooling(prmt_features) # (T, N, L, D) -> (T, N, D)
        return attn_feature.permute(1, 0, 2)             # (T, N, D) -> (N, T, D)

    def tolist(self, paired_wavs: List[Tensor], paired_feature: Tensor):
        assert paired_feature.dim() == 3, "(batch_size, max_seq_len, feat_dim)"
        feature_len = [round(len(wav) / self.downsample_rate) for wav in paired_wavs]
        length_diff = abs(
            paired_feature.size(1)
            - round(max([len(wav) for wav in paired_wavs]) / self.downsample_rate)
        )
        assert (
            length_diff < TOLERABLE_SEQLEN_DIFF
        ), f"{length_diff} >= {TOLERABLE_SEQLEN_DIFF}"
        feature = [f[:l] for f, l in zip(paired_feature, feature_len)]
        return feature

    def forward(
        self,
        paired_wavs: List[Tensor],
        paired_features: Dict[str, Union[Tensor, List[Tensor], Dict[str, Tensor]]],
    ):
        features = self._select_feature(paired_features)
        assert isinstance(features, (list|tuple)), "features should be a list of tensors"
        feature = self._attn_sum(features)

        return self.tolist(paired_wavs, feature)
