import os
import numpy as np
from typing import List, Tuple, Callable, Any, Optional
from functools import wraps
import multiprocessing.dummy as mp
from pathlib import Path
import h5py as h5

import torch
from torch import Tensor

from timeit import default_timer as timer

def timeit(tolerance=0.1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = timer()
            result = func(*args, **kwargs)
            end = timer()
            duration = end - start
            if duration > tolerance:
                print(f'\n[{func.__name__}] - {duration:.2f}s!!')
            return result
        return wrapper
    return decorator

WAIT_FOR_SAVE = 64

class CacheManager:
    def __init__(self,
                 dataset,
                 args,
                 config,
                 process_func: Callable,
                 overwrite_loader: Optional[Callable] = None,
                 use_cache: bool = True,
                 cache_path: Optional[str] = None):
        self.dataset = dataset
        self.args = args
        self.config = config
        self.process_func = process_func
        self.overwrite_loader = overwrite_loader
        self.use_cache = use_cache
        self.cache_path = cache_path

    def __enter__(self):
        if self.use_cache:
            if self.cache_path is None:
                libri_root = self.config['downstream_expert']['datarc']['libri_root']
                upstream_name = self.args.upstream
                dataset_name = self.config['downstream_expert']['datarc']['train'][0]
                layer = str(self.args.upstream_layer_selection)
                self.cache_path = Path(libri_root)/"cache"/upstream_name/dataset_name/f"{layer}.h5"
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

            self.cache_writer = h5.File(self.cache_path, 'a', libver='latest')
            self.cache_reader = self.cache_writer
            self.pool = mp.Pool(1)
            self.saving_features = []

            if not hasattr(self.dataset, '_have_wrapped_loader'):
                self.original_loader = self.dataset._load_wav
                if self.overwrite_loader is not None:
                    self.dataset._load_wav = self.overwrite_loader
                else:
                    self.dataset._load_wav = self.with_cache(self.dataset._load_wav)
                self.dataset._have_wrapped_loader = True

            print(f"[CacheModule] - Use cache at {self.cache_path}")
        else:
            print(f"[CacheModule] - Don't use cache")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'pool') and self.pool:
            self.pool.close()
            self.pool.join()
            del self.pool
        if hasattr(self, 'cache_writer') and self.cache_writer:
            self.cache_writer.close()
            del self.cache_writer
            del self.cache_reader
        if hasattr(self, 'saving_features') and self.saving_features:
            for saving_feature in self.saving_features:
                saving_feature.get()
            del self.saving_features
        if hasattr(self, 'original_loader') and self.original_loader:
            self.dataset._load_wav = self.original_loader
            if hasattr(self.dataset, '_have_wrapped_loader'):
                del self.dataset._have_wrapped_loader

    def _parse_cache_path(self, wavname):
        return wavname

    @timeit(1)
    def _save_cache(self, wavname, feature):
        np_feature = feature.cpu().numpy()
        feature_path = self._parse_cache_path(wavname)
        try:
            self.cache_writer.create_dataset(feature_path, data=np_feature, compression='lzf', shuffle=True)
        except RuntimeError:
            print(f'Failed to save {feature_path}')

    def async_save_caches(self, wavnames: List[str], features: List[Tensor]):
        while len(self.saving_features) > WAIT_FOR_SAVE:
            self.saving_features.pop(0).get()
        for wavname, feature in zip(wavnames, features):
            self.saving_features.append(self.pool.apply_async(self._save_cache, (wavname, feature)))

    @timeit(1)
    def _load_cache(self, wavname):
        cache_path = self._parse_cache_path(wavname)
        if (feature := self.cache_reader.get(cache_path)) is not None:
            return feature[:]
        else:
            return None

    def with_cache(self, func):
        @wraps(func)
        def wrapper(wavpath: str):
            wavname = Path(wavpath).stem
            feature = self._load_cache(wavname)
            return func(wavpath) if feature is None else feature
        return wrapper

    def get_features(self, wavs: List[Tensor], wavnames: List[str], save=True) -> List[Tensor]:
        cached_states = [wav.ndim != 1 for wav in wavs]

        cached_features = []
        uncached_wavs, uncached_names = [], []
        for wav, wavname, cached in zip(wavs, wavnames, cached_states):
            if cached:
                cached_features.append(wav)
            else:
                uncached_wavs.append(wav)
                uncached_names.append(wavname)

        if not self.use_cache:
            assert len(cached_features) == 0, "cached data is not empty when not using cache"

        # process uncached data
        uncached_features = self.process_func(uncached_wavs) if uncached_wavs else []
        if self.use_cache and save and uncached_features:
            self.async_save_caches(uncached_names, uncached_features)

        # merge cached and uncached data
        features = []
        for cached in cached_states:
            features.append(cached_features.pop(0) if cached else uncached_features.pop(0))
        assert not uncached_features and not cached_features

        return features