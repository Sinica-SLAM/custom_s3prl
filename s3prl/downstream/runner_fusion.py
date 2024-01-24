import os
import sys
import math
import glob
import shutil
import random
import tempfile
import importlib
from pathlib import Path
from typing import List
from functools import partial, wraps

import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import is_initialized, get_rank, get_world_size

from s3prl import hub
from s3prl.optimizers import get_optimizer
from s3prl.schedulers import get_scheduler
from s3prl.upstream.featurizer import *
from s3prl.downstream.cache import CacheManager
from s3prl.upstream.fusioner import *
from s3prl.utility.helper import is_leader_process, get_model_state, show, defaultdict


SAMPLE_RATE = 16000


class ModelEntry:
    def __init__(self, model, name, trainable, interfaces):
        self.model = model
        self.name = name
        self.trainable = trainable
        self.interfaces = interfaces


class RunnerFusion():
    """
    Used to handle high-level concepts of a ML experiment
    eg. training loop, evaluation loop, upstream propagation, optimization, logging, checkpoint saving
    """
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.init_ckpt = torch.load(self.args.init_ckpt, map_location='cpu') if self.args.init_ckpt else {}

        self.upstream1, self.upstream2 = self._get_upstream()
        self.ifeaturizer1, self.ifeaturizer2 = self._get_featurizer()
        self.fusioner = self._get_fusioner()
        self.downstream = self._get_downstream()
        self.all_entries = [self.upstream1, self.ifeaturizer1, self.ifeaturizer2, self.fusioner, self.downstream]
        if not self.self_fusion:
            self.all_entries.append(self.upstream2)

        self.cache1_manager, self.cache2_manager = self._get_cache()


    def _load_weight(self, model, name):
        init_weight = self.init_ckpt.get(name)
        if init_weight:
            show(f'[Runner] - Loading {name} weights from the previous experiment')
            model.load_state_dict(init_weight)


    def _init_model(self, model, name, trainable, interfaces=None):
        for interface in interfaces or []:
            assert hasattr(model, interface), interface

        self._load_weight(model, name)

        if is_initialized() and trainable and any((p.requires_grad for p in model.parameters())):
            model = DDP(model, device_ids=[self.args.local_rank], find_unused_parameters=False)
            for interface in interfaces or []:
                setattr(model, interface, getattr(model.module, interface))

        return ModelEntry(model, name, trainable, interfaces)


    def _get_upstream(self):
        if self.args.upstream2 is None:
            self.args.upstream2 = self.args.upstream1
        self.self_fusion = self.args.upstream1 == self.args.upstream2
        if self.self_fusion:
            print(f'[Runner] - Use self fusion , upstream: {self.args.upstream1}')
        else:
            print(f'[Runner] - Use cross fusion , upstream1: {self.args.upstream1}, upstream2: {self.args.upstream2}')

        upstream_refresh = self.args.upstream_refresh

        Upstream1 = getattr(hub, self.args.upstream1)
        ckpt_path1 = self.args.upstream_ckpt1

        if not self.self_fusion:
            Upstream2 = getattr(hub, self.args.upstream2)
            ckpt_path2 = self.args.upstream_ckpt2

        if is_initialized() and get_rank() > 0:
            torch.distributed.barrier()
            upstream_refresh = False

        model1 = Upstream1(
            ckpt = ckpt_path1,
            model_config = None,
            refresh = upstream_refresh,
        ).to(self.args.device)

        if not self.self_fusion:
            model2 = Upstream2(
                ckpt = ckpt_path2,
                model_config = None,
                refresh = upstream_refresh,
            ).to(self.args.device)

        if is_initialized() and get_rank() == 0:
            torch.distributed.barrier()

        if self.args.upstream_trainable1:
            print(f"[Runner] - Upstream1 is trainable")
        else:
            print(f"[Runner] - Upstream1 is not trainable")
        model1 = self._init_model(
            model = model1,
            name = 'Upstream1',
            trainable = self.args.upstream_trainable1,
            interfaces = ["get_downsample_rates"]
        )

        if not self.self_fusion:
            if self.args.upstream_trainable2:
                print(f"[Runner] - Upstream2 is trainable")
            else:
                print(f"[Runner] - Upstream2 is not trainable")
            model2 = self._init_model(
                model = model2,
                name = 'Upstream2',
                trainable = self.args.upstream_trainable2,
                interfaces = ["get_downsample_rates"]
            )
        else:
            model2 = model1

        return model1, model2


    def _get_featurizer(self):
        if self.args.upstream1_feature_selection == 'hidden_states' \
            and self.args.upstream1_layer_selection is not None:
            print(f"[Runner] - iFeaturizer1 is not trainable")
            trainable1 = False
        else:
            print(f"[Runner] - iFeaturizer1 is trainable")
            trainable1 = True
        InnerFeaturizer1 = eval(self.args.ifeaturizer1)
        conf = self.config.get('featurizer_conf', {})
        conf1 = conf.get(self.args.ifeaturizer1, {})
        ifeaturizer1 = InnerFeaturizer1(
            upstream = self.upstream1.model,
            feature_selection = self.args.upstream1_feature_selection,
            layer_selection = self.args.upstream1_layer_selection,
            upstream_device = self.args.device,
            normalize = self.args.upstream_feature_normalize,
            **conf1
        ).to(self.args.device)

        if self.args.upstream2_feature_selection == 'hidden_states' \
            and self.args.upstream2_layer_selection is not None:
            print(f"[Runner] - iFeaturizer2 is not trainable")
            trainable2 = False
        else:
            print(f"[Runner] - iFeaturizer2 is trainable")
            trainable2 = True
        InnerFeaturizer2 = eval(self.args.ifeaturizer2)
        conf2 = conf.get(self.args.ifeaturizer2, {})
        ifeaturizer2 = InnerFeaturizer2(
            upstream = self.upstream1.model if self.self_fusion else self.upstream2.model,
            feature_selection = self.args.upstream2_feature_selection,
            layer_selection = self.args.upstream2_layer_selection,
            upstream_device = self.args.device,
            normalize = self.args.upstream_feature_normalize,
            **conf2
        ).to(self.args.device)

        return self._init_model(
            model = ifeaturizer1,
            name = 'iFeaturizer1',
            trainable = trainable1,
            interfaces = ['output_dim', 'downsample_rate', 'get_feature_lens']
        ), self._init_model(
            model = ifeaturizer2,
            name = 'iFeaturizer2',
            trainable = trainable2,
            interfaces = ['output_dim', 'downsample_rate', 'get_feature_lens']
        )


    def _get_cache(self):
        libri_root = self.config['downstream_expert']['datarc']['libri_root']
        dataset_name = self.config['downstream_expert']['datarc']['train']

        assert os.path.exists(libri_root), f"libri_root {libri_root} does not exist"
        assert len(dataset_name) == 1, f"Only support one dataset for caching, but got {dataset_name}"
        dataset_name = dataset_name[0]

        if not hasattr(self.downstream.model, f"train_dataset"):
            self.downstream.model.get_dataloader("train") # create dataset
        train_dataset = self.downstream.model.train_dataset

        upstream1_name = self.args.upstream1
        layer1 = str(self.args.upstream1_layer_selection)
        process_func1 = partial(self.process_wavs, self.upstream1, self.ifeaturizer1)
        cache1_dir = Path(libri_root)/"cache"/upstream1_name/dataset_name/layer1
        use_cache1 = not self.upstream1.trainable and not self.ifeaturizer1.trainable and self.args.use_cache
        cache1_manager = CacheManager(train_dataset, self.args, self.config, process_func1, self.with_cache, use_cache1, cache1_dir)

        upstream2_name = self.args.upstream2
        layer2 = str(self.args.upstream2_layer_selection)
        process_func2 = partial(self.process_wavs, self.upstream2, self.ifeaturizer2)
        cache2_dir = Path(libri_root)/"cache"/upstream2_name/dataset_name/layer2
        use_cache2 = not self.upstream2.trainable and not self.ifeaturizer2.trainable and self.args.use_cache
        cache2_manager = CacheManager(train_dataset, self.args, self.config, process_func2, self.with_cache, use_cache2, cache2_dir)

        return cache1_manager, cache2_manager


    def _get_fusioner(self):
        Fusioner = eval(self.args.fusioner)
        conf = self.config.get('fusioner_conf')
        if conf is not None:
            conf = conf.get(self.args.fusioner, {})
        fusioner = Fusioner(
            self.ifeaturizer1.model,
            self.ifeaturizer2.model,
            **conf
        ).to(self.args.device)

        return self._init_model(
            model = fusioner,
            name = 'Fusioner',
            trainable = fusioner.trainable,
            interfaces = ['upstream_dim', 'downsample_rate']
        )


    def _get_downstream(self):
        expert = importlib.import_module(f"s3prl.downstream.{self.args.downstream}.expert")
        Downstream = getattr(expert, "DownstreamExpert")

        model = Downstream(
            upstream_dim = self.fusioner.model.upstream_dim,
            upstream_rate = self.fusioner.model.downsample_rate,
            **self.config,
            **vars(self.args)
        ).to(self.args.device)

        return self._init_model(
            model = model,
            name = 'Downstream',
            trainable = True,
            interfaces = ['get_dataloader', 'log_records']
        )


    def _get_optimizer(self, model_params):
        optimizer = get_optimizer(
            model_params,
            self.config['runner']['total_steps'],
            self.config['optimizer']
        )
        self._load_weight(optimizer, 'Optimizer')
        return optimizer


    def _get_scheduler(self, optimizer):
        scheduler = get_scheduler(
            optimizer,
            self.config['runner']['total_steps'],
            self.config['scheduler']
        )
        self._load_weight(scheduler, 'Scheduler')
        return scheduler


    def with_cache(self, func):
        @wraps(func)
        def wrapper(wavpath: str):
            wavname = Path(wavpath).stem
            np_feature1 = self.cache1_manager.load_cache(wavname)
            np_feature2 = self.cache2_manager.load_cache(wavname)
            if not isinstance(np_feature1, np.ndarray) or not isinstance(np_feature2, np.ndarray):
                wav = func(wavpath)
            np_feature1 = np_feature1 if isinstance(np_feature1, np.ndarray) else wav
            np_feature2 = np_feature2 if isinstance(np_feature2, np.ndarray) else wav
            return np_feature1, np_feature2
        return wrapper

    def process_wavs(self, upstream, featurizer, wavs: List[Tensor]) -> List[Tensor]:
        if upstream.trainable:
            features = upstream.model(wavs)
        else:
            with torch.no_grad():
                features = upstream.model(wavs)

        if upstream.trainable or featurizer.trainable:
            features = featurizer.model(wavs, features)
        else:
            with torch.no_grad():
                features = featurizer.model(wavs, features)

        return features


    def train(self):
        # trainable parameters and train/eval mode
        trainable_models = []
        trainable_paras = []
        for entry in self.all_entries:
            if entry.trainable:
                entry.model.train().to(self.args.device)
                trainable_models.append(entry.model)
                trainable_paras += list(entry.model.parameters())
            else:
                entry.model.eval().to(self.args.device)

        # optimizer
        optimizer = self._get_optimizer(trainable_models)

        # scheduler
        scheduler = None
        if self.config.get('scheduler'):
            scheduler = self._get_scheduler(optimizer)

        # specaug
        specaug = None
        if self.config.get('specaug'):
            from .specaug import SpecAug
            specaug = SpecAug(**self.config["specaug"])

        # progress bar
        tqdm_file = sys.stderr if is_leader_process() else open(os.devnull, 'w')
        pbar = tqdm(total=self.config['runner']['total_steps'], dynamic_ncols=True, desc='overall', file=tqdm_file)
        init_step = self.init_ckpt.get('Step')
        if init_step:
            pbar.n = init_step

        # Tensorboard logging
        if is_leader_process():
            logger = SummaryWriter(self.args.expdir)

        batch_ids = []
        backward_steps = 0
        records = defaultdict(list)
        epoch = self.init_ckpt.get('Epoch', 0)
        train_split = self.config['runner'].get("train_dataloader", "train")

        with self.cache1_manager as cache1_manager, self.cache2_manager as cache2_manager:
            while pbar.n < pbar.total:
                try:
                    dataloader = self.downstream.model.get_dataloader(train_split, epoch=epoch)
                except TypeError as e:
                    if "unexpected keyword argument 'epoch'" in str(e):
                        dataloader = self.downstream.model.get_dataloader(train_split)
                        if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, DistributedSampler):
                            dataloader.sampler.set_epoch(epoch)
                    else:
                        raise

                for batch_id, (np_wavs, labels, wavnames) in enumerate(tqdm(dataloader, dynamic_ncols=True, desc='train', file=tqdm_file)):
                    # try/except block for forward/backward
                    try:
                        if pbar.n >= pbar.total:
                            break
                        global_step = pbar.n + 1

                        wavs1, wavs2 = [], []
                        for np_wav1, np_wav2 in np_wavs:
                            wav1 = torch.FloatTensor(np_wav1).to(self.args.device, non_blocking=True)
                            wav2 = torch.FloatTensor(np_wav2).to(self.args.device, non_blocking=True) \
                                if np_wav1 is not np_wav2 else wav1
                            wavs1.append(wav1)
                            wavs2.append(wav2)


                        features1 = cache1_manager.get_features(wavs1, wavnames)
                        features2 = cache2_manager.get_features(wavs2, wavnames)

                        for i, (f1, f2) in enumerate(zip(features1, features2)):
                            if f1.shape != f2.shape:
                                print(f'[Runner] - Warning: feature1.shape: {f1.shape}, feature2.shape: {f2.shape} unmatch!!')
                                f_len_min = min(f1.size(0), f2.size(0))
                                features1[i] = f1.narrow(0, 0, f_len_min)
                                features2[i] = f2.narrow(0, 0, f_len_min)


                        features = self.fusioner.model(features1, features2)

                        if specaug:
                            features, _ = specaug(features)

                        loss = self.downstream.model(
                            train_split,
                            features, labels, wavnames,
                            records = records,
                        )
                        batch_ids.append(batch_id)

                        gradient_accumulate_steps = self.config['runner'].get('gradient_accumulate_steps')
                        (loss / gradient_accumulate_steps).backward()
                        del loss

                    except RuntimeError as e:
                        if 'CUDA out of memory' in str(e):
                            print(f'[Runner] - CUDA out of memory at step {global_step}')
                            if is_initialized():
                                raise
                            with torch.cuda.device(self.args.device):
                                torch.cuda.empty_cache()
                            optimizer.zero_grad()
                            continue
                        else:
                            raise

                    # whether to accumulate gradient
                    backward_steps += 1
                    if backward_steps % gradient_accumulate_steps > 0:
                        continue

                    # gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_paras, self.config['runner']['gradient_clipping'])

                    # optimize
                    if math.isnan(grad_norm):
                        print(f'[Runner] - grad norm is NaN at step {global_step}')
                    else:
                        optimizer.step()
                        if hasattr(self.ifeaturizer1.model, "step"):
                            self.ifeaturizer1.model.step()
                        if hasattr(self.ifeaturizer2.model, "step"):
                            self.ifeaturizer2.model.step()
                    optimizer.zero_grad()

                    # adjust learning rate
                    if scheduler:
                        scheduler.step()

                    if not is_leader_process():
                        batch_ids = []
                        records = defaultdict(list)
                        continue

                    # logging
                    if global_step % self.config['runner']['log_step'] == 0:
                        self.downstream.model.log_records(
                            train_split,
                            records = records,
                            logger = logger,
                            global_step = global_step,
                            batch_ids = batch_ids,
                            total_batch_num = len(dataloader),
                        )
                        batch_ids = []
                        records = defaultdict(list)
                        if hasattr(self.ifeaturizer1.model, "show"):
                            self.ifeaturizer1.model.show()
                        if hasattr(self.ifeaturizer2.model, "show"):
                            self.ifeaturizer2.model.show()

                    # evaluation and save checkpoint
                    save_names = []

                    if global_step % self.config['runner']['eval_step'] == 0:
                        for split in self.config['runner']['eval_dataloaders']:
                            save_names += self.evaluate(split, logger, global_step)

                    if global_step % self.config['runner']['save_step'] == 0:
                        def check_ckpt_num(directory):
                            max_keep = self.config['runner']['max_keep']
                            ckpt_pths = glob.glob(f'{directory}/states-*.ckpt')
                            if len(ckpt_pths) >= max_keep:
                                ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
                                for ckpt_pth in ckpt_pths[:len(ckpt_pths) - max_keep + 1]:
                                    os.remove(ckpt_pth)
                        check_ckpt_num(self.args.expdir)
                        save_names.append(f'states-{global_step}.ckpt')

                    if len(save_names) > 0:
                        all_states = {
                            'Optimizer': optimizer.state_dict(),
                            'Step': global_step,
                            'Epoch': epoch,
                            'Args': self.args,
                            'Config': self.config,
                        }

                        for entry in self.all_entries:
                            if entry.trainable:
                                all_states[entry.name] = get_model_state(entry.model)

                        if scheduler:
                            all_states['Scheduler'] = scheduler.state_dict()

                        if is_initialized():
                            all_states['WorldSize'] = get_world_size()

                        save_paths = [os.path.join(self.args.expdir, name) for name in save_names]
                        tqdm.write(f'[Runner] - Save the checkpoint to:')
                        for i, path in enumerate(save_paths):
                            tqdm.write(f'{i + 1}. {path}')
                            torch.save(all_states, path)

                    pbar.update(1)
                epoch += 1

        pbar.close()

        if is_leader_process():
            logger.close()


    def evaluate(self, split=None, logger=None, global_step=0):
        """evaluate function will always be called on a single process even during distributed training"""

        # When this member function is called directly by command line
        not_during_training = split is None and logger is None and global_step == 0
        if not_during_training:
            split = self.args.evaluate_split
            tempdir = tempfile.mkdtemp()
            logger = SummaryWriter(tempdir)

        # fix seed to guarantee the same evaluation protocol across steps 
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
            with torch.cuda.device(self.args.device):
                torch.cuda.empty_cache()

        # record original train/eval states and set all models to eval
        trainings = []
        for entry in self.all_entries:
            trainings.append(entry.model.training)
            entry.model.eval().to(self.args.device)

        # prepare data
        dataloader = self.downstream.model.get_dataloader(split)
        evaluate_ratio = float(self.config["runner"].get("evaluate_ratio", 1))
        evaluate_steps = round(len(dataloader) * evaluate_ratio)

        batch_ids = []
        records = defaultdict(list)
        for batch_id, (wavs, *other) in enumerate(tqdm(dataloader, dynamic_ncols=True, desc=split, total=evaluate_steps)):
            if batch_id > evaluate_steps:
                break

            with torch.no_grad():
                wavs = [torch.FloatTensor(wav).to(self.args.device, non_blocking=True) for wav in wavs]
                features1 = self.process_wavs(self.upstream1, self.ifeaturizer1, wavs)
                features2 = self.process_wavs(self.upstream2, self.ifeaturizer2, wavs)
                features = self.fusioner.model(features1, features2)
                self.downstream.model(
                    split,
                    features, *other,
                    records = records,
                    batch_id = batch_id,
                )
                batch_ids.append(batch_id)

        save_names = self.downstream.model.log_records(
            split,
            records = records,
            logger = logger,
            global_step = global_step,
            batch_ids = batch_ids,
            total_batch_num = len(dataloader),
        )
        batch_ids = []
        records = defaultdict(list)

        # prepare back to training
        if torch.cuda.is_available():
            with torch.cuda.device(self.args.device):
                torch.cuda.empty_cache()

        for entry, training in zip(self.all_entries, trainings):
            if training:
                entry.model.train().to(self.args.device)

        if not_during_training:
            logger.close()
            shutil.rmtree(tempdir)

        return [] if type(save_names) is not list else save_names

    def inference(self):
        filepath = Path(self.args.evaluate_split)
        assert filepath.is_file(), filepath
        filename = filepath.stem

        if hasattr(self.downstream.model, "load_audio"):
            wav = self.downstream.model.load_audio(filepath)
        else:
            wav, sr = torchaudio.load(str(filepath))
            assert sr == SAMPLE_RATE, sr
        wavs = [wav.view(-1).to(self.args.device)]

        for entry in self.all_entries:
            entry.model.eval().to(self.args.device)

        with torch.no_grad():
            features1 = self.process_wavs(self.upstream1, self.ifeaturizer1, wavs)
            features2 = self.process_wavs(self.upstream2, self.ifeaturizer2, wavs)
            features = self.fusioner.model(features1, features2)
            self.downstream.model.inference(features, [filename])
