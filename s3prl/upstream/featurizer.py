import sys
from typing import Callable, Dict, List, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from torch.nn.functional import gumbel_softmax

from s3prl.utility.helper import show

from s3prl.upstream.interfaces import UpstreamBase, SAMPLE_RATE


def tolist(f_lens: List[int], features: Tensor):
    assert features.dim() == 3, "(batch_size, max_seq_len, feat_dim)"
    feature = [f[:l] for f, l in zip(features, f_lens)]
    return feature

def padding(features: List[Tensor], padding_value: float = 0) -> Tuple[List[int], Tensor]:
    feature_lens = [len(f) for f in features]
    feature = pad_sequence(features, batch_first=True, padding_value=padding_value)
    return feature_lens, feature


class Featurizer(nn.Module):
    def __init__(
        self,
        upstream: UpstreamBase,
        feature_selection: str = "hidden_states",
        upstream_device: str = "cuda",
        layer_selection: int = None,
        normalize: bool = False,
        **kwargs,
    ):
        super().__init__()

        upstream.eval()
        paired_wavs = [torch.randn(SAMPLE_RATE).to(upstream_device)]
        with torch.no_grad():
            paired_features = upstream(paired_wavs)

        if feature_selection not in paired_features:
            if "hidden_states" in paired_features:
                show(
                    f"[{self.__class__.__name__}] - Warning: {feature_selection} is not a supported args.upstream_feature_selection."
                    f' Using "hidden_states" as the default key.',
                    file=sys.stderr,
                )
                feature_selection = "hidden_states"
            else:
                show(
                    f"[{self.__class__.__name__}] - Error: {feature_selection} is not a supported args.upstream_feature_selection."
                    f' The default key "hidden_states" is also not supported.'
                    f" Please specify -s with the following options: {list(paired_wavs.keys())}",
                    file=sys.stderr,
                )
                raise ValueError
        self.feature_selection = feature_selection
        self.layer_selection = layer_selection
        self.normalize = normalize

        feature = self._select_feature(paired_features)
        if isinstance(feature, (list, tuple)):
            self.layer_num = len(feature)
            show(
                f"[{self.__class__.__name__}] - Take a list of {self.layer_num} features and weighted sum them.",
                file=sys.stderr,
            )
            self.weights = nn.Parameter(torch.zeros(self.layer_num))
            feature = self._weighted_sum([f.cpu() for f in feature])
        else:
            feature = feature.cpu()

        self.output_dim = feature.size(-1)
        if hasattr(upstream, "get_downsample_rates"):
            self.downsample_rate = upstream.get_downsample_rates(feature_selection)
            show(
                f"[{self.__class__.__name__}] - The selected feature {feature_selection}'s downsample rate is {self.downsample_rate}",
                file=sys.stderr,
            )
        else:
            self.downsample_rate = round(
                max(len(wav) for wav in paired_wavs) / feature.size(1)
            )
            show(
                f"[{self.__class__.__name__}] - Warning: The provided upstream does not give statis downsample rate"
                ' by the "get_downsample_rates" interface (see upstream/example/expert.py).'
                " The downsample rate is calculated dynamically basing on the shape of the"
                f" input waveforms v.s. the output features: {self.downsample_rate}",
                file=sys.stderr,
            )

    def _select_feature(self, features):
        feature = features.get(self.feature_selection)

        if isinstance(feature, dict):
            feature = list(feature.values())

        if isinstance(feature, (list, tuple)) and len(feature) == 1:
            feature = feature[0]

        if isinstance(feature, (list, tuple)) and isinstance(self.layer_selection, int):
            feature = feature[self.layer_selection]

        return feature

    def _weighted_sum(self, feature):
        stacked_feature = torch.stack(feature, dim=0)

        if self.normalize:
            stacked_feature = F.layer_norm(
                stacked_feature, (stacked_feature.shape[-1],)
            )

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(self.layer_num, -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature

    def get_feature_lens(self, wavs: List[Tensor]) -> List[int]:
        def get_feature_len(wav: Tensor) -> int:
            # the length is the integer smaller but closest to the ratio, even if exact division
            return -(-(len(wav)) // self.downsample_rate) - 1
        return [get_feature_len(wav) for wav in wavs]

    def forward(
        self,
        paired_wavs: List[Tensor],
        paired_features: Dict[str, Union[Tensor, List[Tensor], Dict[str, Tensor]]],
    ) -> Tensor:
        features = self._select_feature(paired_features)
        if isinstance(features, (list, tuple)):
            features = self._weighted_sum(features)

        f_lens = self.get_feature_lens(paired_wavs)
        return tolist(f_lens, features)


class AnnealSoftmax(Featurizer):
    def __init__(self, upstream: UpstreamBase, feature_selection: str = "hidden_states", upstream_device: str = "cuda", layer_selection: int = None, normalize: bool = False, **kwargs):
        super().__init__(upstream, feature_selection, upstream_device, layer_selection, normalize, **kwargs)

        self.init_temp(**kwargs)

        assert feature_selection == "hidden_states" and layer_selection is None, f"{self.__class__.__name__} only support hidden_states feature selection and layer_selection is None"

    def init_temp(self, initT = 1.0, turnT = 0.1, finalT = 0.0001, linear_num = 30000, exp_period = 30000):
        self.initT = initT
        self.turnT = turnT
        self.finalT = finalT
        self.linear_num = linear_num
        self.exp_period = exp_period

        self.linear_step = (self.initT - self.turnT) / self.linear_num
        self.exp_factor = math.pow(self.finalT / self.turnT, 1 / self.exp_period)

        self.temp = nn.parameter.Parameter(torch.tensor(self.initT), requires_grad=False)

    def show(self):
        print(f"[{self.__class__.__name__}] - temp: {self.temp.item():.4f}")

    def _get_train_norm_weights(self):
        # in training mode, use softmax
        return F.softmax(self.weights/self.temp, dim=-1) # (L)

    def _get_eval_norm_weights(self):
        # in eval mode, use the argmax of weights
        _, max_idx = self.weights.max(dim=-1)
        return F.one_hot(max_idx, num_classes=self.layer_num).float() # (L)

    def _get_norm_weights(self):
        if self.training:
            return self._get_train_norm_weights()
        else:
            return self._get_eval_norm_weights()

    def _weighted_sum(self, feature): # [L] (B, T, D)
        stacked_feature = torch.stack(feature, dim=0) # (L, B, T, D)

        if self.normalize: # apply on the last dimension (D)
            stacked_feature = F.layer_norm(stacked_feature, (stacked_feature.shape[-1],)) # (L, B, T, D)

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(self.layer_num, -1) # (L, B*T*D)
        norm_weights = self._get_norm_weights() # (L)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0) # (B*T*D)
        weighted_feature = weighted_feature.view(*origin_shape) # (B, T, D)

        return weighted_feature # (B, T, D)

    def step(self):
        temp = self.temp.item()

        temp = temp-self.linear_step if temp > self.turnT else temp*self.exp_factor
        temp = max(temp, self.finalT)

        with torch.no_grad():
            self.temp.fill_(temp)


class GumbelSoftmax(AnnealSoftmax):
    def _get_train_norm_weights(self):
        # in training mode, use the gumbel softmax
        log_probs = F.log_softmax(self.weights/self.temp, dim=-1) # (L)
        gumbel_select = gumbel_softmax(log_probs, hard=True, dim=-1) # (L)
        return gumbel_select # (L)


class GumbelSoftmax2(AnnealSoftmax):
    def _get_train_norm_weights(self):
        # in training mode, use the gumbel softmax
        return gumbel_softmax(self.weights, tau=self.temp, hard=True, dim=-1) # (L)



class AnnealFusion(AnnealSoftmax):
    def __init__(
        self,
        upstream: UpstreamBase,
        feature_selection: str = "hidden_states",
        upstream_device: str = "cuda",
        layer_selection: int = None,
        normalize: bool = False,
        **kwargs,
    ):
        super(Featurizer, self).__init__()

        upstream.eval()
        paired_wavs = [torch.randn(SAMPLE_RATE).to(upstream_device)]
        with torch.no_grad():
            paired_features = upstream(paired_wavs)

        assert feature_selection is None or feature_selection == "hidden_states", f"{self.__class__.__name__} only support hidden_states feature selection"
        assert layer_selection is None, f"{self.__class__.__name__} only support layer_selection is None"
        self.feature_selection = "hidden_states"
        self.layer_selection = None
        self.normalize = normalize

        features = self._select_feature(paired_features)
        self.layer_num = len(features)
        assert all(f.size(-1) == features[0].size(-1) for f in features), f"{self.__class__.__name__} only support the same feature dimension"
        self.output_dim = features[0].size(-1)

        if hasattr(upstream, "get_downsample_rates"):
            self.downsample_rate = upstream.get_downsample_rates(feature_selection)
            show(
                f"[{self.__class__.__name__}] - The selected feature {feature_selection}'s downsample rate is {self.downsample_rate}",
                file=sys.stderr,
            )
        else:
            self.downsample_rate = round(
                max(len(wav) for wav in paired_wavs) / features[0].size(1)
            )
            show(
                f"[{self.__class__.__name__}] - Warning: The provided upstream does not give statis downsample rate"
                ' by the "get_downsample_rates" interface (see upstream/example/expert.py).'
                " The downsample rate is calculated dynamically basing on the shape of the"
                f" input waveforms v.s. the output features: {self.downsample_rate}",
                file=sys.stderr,
            )

        self.init_temp(**kwargs)
        self.weights = nn.parameter.Parameter(torch.zeros(self.layer_num, self.output_dim))

    def _weighted_sum(self, features): # [L] (B, T, D)
        stacked_feature = torch.stack(features, dim=0) # (L, B, T, D)
        L, B, T, D = stacked_feature.shape

        if self.normalize: # apply on the last dimension (D)
            stacked_feature = F.layer_norm(stacked_feature, (D,))

        stacked_feature = stacked_feature.view(L, -1, D) # (L, B*T, D)
        norm_weights = self._get_norm_weights().unsqueeze(1) # (L, 1, D)
        weighted_feature = (norm_weights * stacked_feature).sum(dim=0) # (B*T, D)
        weighted_feature = weighted_feature.view(B, T, D) # (B, T, D)

        return weighted_feature # (B, T, D)

    def _get_train_norm_weights(self):
        # in training mode, use softmax
        return F.softmax(self.weights/self.temp, dim=0) # (L, D)

    def _get_eval_norm_weights(self):
        # in eval mode, use the argmax of weights
        _, max_idx = self.weights.max(dim=0) # (D)
        return F.one_hot(max_idx, num_classes=self.layer_num).T.float() # (L, D)

    def _get_norm_weights(self):
        if self.training:
            return self._get_train_norm_weights()
        else:
            return self._get_eval_norm_weights()


class GumbelFusion(AnnealFusion):
    def _get_train_norm_weights(self):
        # in training mode, use the gumbel softmax
        log_probs = F.log_softmax(self.weights/self.temp, dim=0) # (L, D)
        gumbel_select = gumbel_softmax(log_probs, hard=True, dim=0) # (L, D)
        return gumbel_select # (L, D)


class GumbelFusion2(AnnealFusion):
    def _get_train_norm_weights(self):
        # in training mode, use the gumbel softmax
        return gumbel_softmax(self.weights, tau=self.temp, hard=True, dim=0) # (L, D)

