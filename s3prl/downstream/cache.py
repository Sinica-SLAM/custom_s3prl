import os
import numpy as np
from typing import List, Tuple, Callable, Any
from functools import wraps
import multiprocessing.dummy as mp
from pathlib import Path
import h5py as h5

import torch
from torch import nn
from torch import Tensor


class CacheModule:
    def __init__(self,
                 process_func: Callable,
                 cache_path: str,
                 device: str,
                 num_worker: int = None,
                 sep: str = '-'):
        self.process_func = process_func
        self.cache_path = Path(cache_path)
        self.device = device
        self.num_worker = num_worker or os.cpu_count()
        self.sep = sep

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        assert len(self.sep) == 1, f'seq must be a single character, got {self.sep}'

        self.cache_writer = h5.File(self.cache_path, 'a', libver='latest')
        self.cache_reader = h5.File(self.cache_path, 'r', libver='latest', swmr=True)
        self.pool = mp.Pool(self.num_worker)
        self.saving_features = None

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, 'pool') and self.pool:
            self.pool.close()
            self.pool.join()
            del self.pool
        if hasattr(self, 'cache_writer') and self.cache_writer:
            self.cache_writer.close()
            del self.cache_writer
        if hasattr(self, 'cache_reader') and self.cache_reader:
            self.cache_reader.close()
            del self.cache_reader
        if hasattr(self, 'saving_features') and self.saving_features:
            self.saving_features.get()
            del self.saving_features

    def _parse_cache_path(self, wavname):
        return wavname.replace(self.sep, '/')

    def have_cached(self, wavname):
        return self._parse_cache_path(wavname) in self.cache_reader

    def _save_cache(self, wavname, feature):
        if not self.have_cached(wavname):
            np_feature = feature.cpu().numpy()
            feature_path = self._parse_cache_path(wavname)
            self.cache_writer.create_dataset(feature_path, data=np_feature, compression='lzf', shuffle=True)

    def async_save_caches(self, wavnames: List[str], features: List[Tensor]):
        if self.saving_features:
            self.saving_features.get()
        self.saving_features = self.pool.starmap_async(self._save_cache, zip(wavnames, features))

    def _load_cache(self, wavname):
        cache_path = self._parse_cache_path(wavname)
        return torch.from_numpy(self.cache_reader[cache_path][:])

    def with_cache(self, func):
        @wraps(func)
        def wrapper(wavpath: str):
            wavname = Path(wavpath).stem
            return self._load_cache(wavname) \
                if self.have_cached(wavname) \
                else func(wavpath)
        return wrapper

    def _np_to_device(self, array, *args, **kwargs):
         return torch.FloatTensor(array).to(self.device, *args, **kwargs)

    def get_features(self, wavs, labels, wavnames, save=True):
        # use cache data if have cached
        uncached_datas, cached_datas = [], []
        for data in zip(wavs, labels, wavnames):
            (uncached_datas, cached_datas)[data[0].ndim != 1].append(data) # wav is 1-dim

        # move uncached wavs to device
        uncached_features, uncached_labels, uncached_names = [], [], []
        if uncached_datas:
            uncached_wavs, uncached_labels, uncached_names = zip(*uncached_datas)
            uncached_wavs = [self._np_to_device(w) for w in uncached_wavs]

        # move cached features to device
        cached_features, cached_labels, cached_names = [], [], []
        if cached_datas:
            cached_features, cached_labels, cached_names = zip(*cached_datas)
            cached_features = [self._np_to_device(f, non_blocking=True) for f in cached_features]

        # process uncached data
        if uncached_datas:
            uncached_features = self.process_func(uncached_wavs)
            if save:
                self.async_save_caches(uncached_names, uncached_features)

        # merge cached and uncached data
        features = list(cached_features) + list(uncached_features)
        labels   = list(cached_labels)   + list(uncached_labels)
        wavnames = list(cached_names)    + list(uncached_names)

        return features, labels, wavnames