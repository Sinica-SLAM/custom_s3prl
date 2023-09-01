import os
import numpy as np
from typing import List, Tuple, Callable, Any, Optional, Dict
from functools import wraps
import multiprocessing.dummy as mp
from pathlib import Path
import random

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
                print(f'[{func.__name__}] - {duration:.2f}s!!')
            return result
        return wrapper
    return decorator

WAIT_FOR_SAVE = 128

class CacheManager:
    def __init__(self,
                 dataset,
                 args,
                 config,
                 process_func: Callable,
                 load_wrapper: Optional[Callable] = None,
                 use_cache: bool = True,
                 cache_dir: Optional[str] = None):
        self.dataset = dataset
        self.args = args
        self.config = config
        self.process_func = process_func
        self.load_wrapper = load_wrapper or self.with_cache
        self.use_cache = use_cache
        self.cache_in_ram = args.cache_ram_ratio is not None and self.use_cache
        self.cache_in_disk = self.use_cache
        self.cache_dir = cache_dir or self._get_default_cache_dir()

        if self.cache_in_ram:
            assert self.config['downstream_expert']['datarc']['num_workers'] <= 1, \
                "num_workers must not greater than 2 when cache in ram"

    def __enter__(self):
        if not hasattr(self.dataset, '_have_wrapped_loader'):
            self.original_loader = self.dataset._load_wav
            self.dataset._load_wav = self.load_wrapper(self.original_loader)
            self.dataset._have_wrapped_loader = True
            print(f"[CacheModule] - Wrap loader with {self.load_wrapper.__name__}")

        if not self.use_cache:
            print(f"[CacheModule] - Don't use cache")
            return self

        self.saving_features = []
        self.pool = mp.Pool(min(4, os.cpu_count()), initializer=random.seed, initargs=(os.getpid(),))

        if self.cache_in_ram:
            self.cache_ratio = self.args.cache_ram_ratio
            self.cache_ram: Dict[str, np.ndarray] = {}
            print(f"[CacheModule] - Use cache in RAM with ratio {self.cache_ratio:.2f}")

        if self.cache_in_disk:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"[CacheModule] - Use cache at {self.cache_dir}")

        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'saving_features') and self.saving_features:
            for saving_feature in self.saving_features:
                saving_feature.get()
            del self.saving_features
        if hasattr(self, 'pool') and self.pool:
            self.pool.close()
            self.pool.join()
            del self.pool
        if hasattr(self, 'original_loader') and self.original_loader:
            self.dataset._load_wav = self.original_loader
            if hasattr(self.dataset, '_have_wrapped_loader'):
                del self.dataset._have_wrapped_loader

    def _get_default_cache_dir(self):
            libri_root = self.config['downstream_expert']['datarc']['libri_root']
            upstream_name = self.args.upstream
            dataset_name = self.config['downstream_expert']['datarc']['train'][0]
            layer = str(self.args.upstream_layer_selection)
            return Path(libri_root)/"cache"/upstream_name/dataset_name/layer

    def _parse_cache_name(self, wavname: str):
        return wavname.replace('-', '/')

    def check_in_ram(self, cache_name: str):
        return cache_name in self.cache_ram

    def _save_cache_to_ram(self, cache_name: str, np_feature: np.ndarray):
        if not self.check_in_ram(cache_name):
            if random.random() >= self.cache_ratio:
                self.cache_ram[cache_name] = None
                return False
            else:
                self.cache_ram[cache_name] = np_feature
        return True

    @timeit(2)
    def _save_cache_to_file(self, cache_name: str, np_feature: np.ndarray):
        cache_path = self.cache_dir/f"{cache_name}.npy"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, np_feature)

    def save_cache(self, wavname: str, feature: Tensor):
        np_feature = feature.detach().cpu().numpy()
        cache_name = self._parse_cache_name(wavname)
        try:
            if self.cache_in_ram:
                self._save_cache_to_ram(cache_name, np_feature)
            if self.cache_in_disk:
                self._save_cache_to_file(cache_name, np_feature)
        except RuntimeError:
            print(f'Failed to save {cache_name}')
        return None

    @timeit(3)
    def save_cache_batch(self, wavnames: List[str], features: List[Tensor]):
        self.saving_features = [f for f in self.saving_features if not f.ready() or f.get()]
        while len(self.saving_features) > WAIT_FOR_SAVE:
            self.saving_features.pop(0).get()

        for wavname, feature in zip(wavnames, features):
            self.saving_features.append(self.pool.apply_async(self.save_cache, (wavname, feature)))

    def _load_cache_from_ram(self, cache_name: str):
        return self.cache_ram.get(cache_name)

    @timeit(3)
    def _load_cache_from_file(self, cache_name: str):
        cache_path = self.cache_dir/f"{cache_name}.npy"
        try:
            return np.load(cache_path)
        except FileNotFoundError:
            pass
        except Exception:
            print(f'Failed to load {cache_path}')
            cache_path.unlink()
        return None

    def load_cache(self, wavname):
        if not self.use_cache:
            return None

        cache_name = self._parse_cache_name(wavname)

        if self.cache_in_ram:
            if (np_feature := self._load_cache_from_ram(cache_name)) is not None:
                return np_feature

        if self.cache_in_disk:
            if (np_feature := self._load_cache_from_file(cache_name)) is not None:
                if self.cache_in_ram:
                    self._save_cache_to_ram(cache_name, np_feature)
                return np_feature

        return None

    def with_cache(self, func):
        @wraps(func)
        def wrapper(wavpath: str):
            wavname = Path(wavpath).stem
            np_feature = self.load_cache(wavname)
            return func(wavpath) if np_feature is None else np_feature
        return wrapper if self.use_cache else func

    @timeit(1)
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
            assert len(cached_features) == 0, "cached data is not empty when not using cache, something wrong"

        # process uncached data
        uncached_features = self.process_func(uncached_wavs) if uncached_wavs else []
        if self.use_cache and save and uncached_features:
            self.save_cache_batch(uncached_names, uncached_features)

        # merge cached and uncached data
        features = []
        for cached in cached_states:
            features.append(cached_features.pop(0) if cached else uncached_features.pop(0))
        assert not uncached_features and not cached_features

        return features