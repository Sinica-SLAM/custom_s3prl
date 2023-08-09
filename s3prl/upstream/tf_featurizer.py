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
        self.layer_selection = layer_selection
        self.normalize = normalize
        self.downsample_rate = upstream.get_downsample_rates()
        self.output_dim = upstream.hidden_size
        self.upstream_device = upstream_device

        self.multihead_attn = MultiheadAttention(embed_dim=upstream.hidden_size,
                                                    num_heads=4,
                                                    bias=False,
                                                    batch_first=True)

    def _select_feature(self, features):
        feature = features.get("hidden_states")

        if isinstance(feature, dict):
            feature = list(feature.values())

        return feature

    def _tf_select(self, features: List[Tensor]):
        """select tenth vector after self-attention as the feature"""
        features = torch.stack(features, dim=0) # L*(N, T, D) -> (L, N, T, D)
        if self.normalize:
            features = F.layer_norm(features, features.shape[3:]) # (L, N, T, D)
        L, N, T, D = features.shape
        features = features.permute(2, 1, 0, 3) # (L, N, T, D) -> (T, N, L, D)
        features = features.view(-1, L, D) # (T, N, L, D) -> (T*N, L, D)
        attn_output, _ = self.multihead_attn(
            features,
            features,
            features,
            need_weights=False
        ) # (T*N, L, D)
        attn_output = attn_output.view(T, N, L, D) # (T*N, L, D) -> (T, N, L, D)
        attn_output = attn_output.permute(2, 1, 0, 3) # (T, N, L, D) -> (L, N, T, D)
        selected_feature = attn_output[self.layer_selection] # (L, N, T, D) -> (N, T, D)
        return selected_feature

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
        feature = self._tf_select(features)

        return self.tolist(paired_wavs, feature)
