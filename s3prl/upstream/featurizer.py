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

        self.initT = kwargs.get("initT", 1.0)
        self.turnT = kwargs.get("turnT", 0.1)
        self.finalT = kwargs.get("finalT", 0.0001)

        self.linear_num = kwargs.get("linear_num", 30000)
        self.exp_period = kwargs.get("exp_period", 30000)

        self.linear_step = (self.initT - self.turnT) / self.linear_num
        self.exp_factor = math.pow(self.finalT / self.turnT, 1 / self.exp_period)

        self.temp = nn.parameter.Parameter(torch.tensor(self.initT), requires_grad=False)

        assert feature_selection == "hidden_states" and layer_selection is None, f"{self.__class__.__name__} only support hidden_states feature selection and layer_selection is None"

        self._weighted_sum = self._auto_select

        self.show()

    def show(self):
        print(f"[{self.__class__.__name__}] - temp: {self.temp.item():.4f}")

    def _get_norm_weights(self):
        return F.softmax(self.weights/self.temp, dim=-1)

    def _auto_select(self, feature):
        stacked_feature = torch.stack(feature, dim=0)

        if self.normalize:
            stacked_feature = F.layer_norm(
                stacked_feature, (stacked_feature.shape[-1],)
            )

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(self.layer_num, -1)
        norm_weights = self._get_norm_weights()
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature

    def step(self):
        temp = self.temp.item()

        temp = temp-self.linear_step if temp > self.turnT else temp*self.exp_factor
        temp = max(temp, self.finalT)

        with torch.no_grad():
            self.temp.fill_(temp)


class GumbelSoftmax(AnnealSoftmax):
    def __init__(self, upstream: UpstreamBase, feature_selection: str = "hidden_states", upstream_device: str = "cuda", layer_selection: int = None, normalize: bool = False, **kwargs):
        super().__init__(upstream, feature_selection, upstream_device, layer_selection, normalize, **kwargs)

    def _get_norm_weights(self):
        log_probs = F.log_softmax(self.weights/self.temp, dim=-1)
        return gumbel_softmax(log_probs, hard=True, dim=-1)
