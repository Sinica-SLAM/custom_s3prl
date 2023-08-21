# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ the phone dataset ]
#   Author       [ S3PRL, Xuankai Chang ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import logging
import os
import random
#-------------#
import pandas as pd
from tqdm import tqdm
from pathlib import Path
#-------------#
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
#-------------#
import torchaudio
#-------------#
from .dictionary import Dictionary

SAMPLE_RATE = 16000
HALF_BATCHSIZE_TIME = 2000


####################
# Sequence Dataset #
####################
class SequenceDataset(Dataset):

    def __init__(self, split, bucket_size, dictionary, kaldi_root, bucket_file, **kwargs):
        super(SequenceDataset, self).__init__()

        self.dictionary = dictionary
        self.kaldi_root = kaldi_root
        self.sample_rate = SAMPLE_RATE
        self.split_sets = kwargs[split]

        # Read table for bucketing
        assert os.path.isdir(bucket_file), 'Please first run `python3 preprocess/generate_len_for_bucket.py -h` to get bucket file.'

        # Wavs
        table_list = []
        for item in self.split_sets:
            file_path = os.path.join(bucket_file, item + ".csv")
            if os.path.exists(file_path):
                table_list.append(
                    pd.read_csv(file_path)
                )
            else:
                logging.warning(f'{item} is not found in bucket_file: {bucket_file}, skipping it.')

        table_list = pd.concat(table_list).set_index('id')
        table_list = table_list.sort_values(by=['length'], ascending=False)

        X = table_list['file_path'].tolist()
        X_lens = table_list['length'].tolist()

        assert len(X) != 0, f"0 data found for {split}"

        # Transcripts
        Y = self._load_transcript(table_list)

        x_names = set([self._parse_x_name(x) for x in X])
        y_names = set(Y.keys())
        usage_list = list(x_names & y_names)

        Y = {key: Y[key] for key in usage_list}

        self.Y = {
            k: self.dictionary.encode_line(
                v, line_tokenizer=lambda x: x.split()
            ).long()
            for k, v in Y.items()
        }

        # Use bucketing to allow different batch sizes at run time
        self.X = []
        batch_x, batch_len = [], []

        for x, x_len in tqdm(zip(X, X_lens), total=len(X), desc=f'ASR dataset {split}', dynamic_ncols=True):
            if self._parse_x_name(x) in usage_list:
                batch_x.append(x)
                batch_len.append(x_len)

                # Fill in batch_x until batch is full
                if len(batch_x) == bucket_size:
                    # Half the batch size if seq too long
                    if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME):
                        self.X.append(batch_x[:bucket_size//2])
                        self.X.append(batch_x[bucket_size//2:])
                    else:
                        self.X.append(batch_x)
                    batch_x, batch_len = [], []

        # Gather the last batch
        if len(batch_x) > 1:
            if self._parse_x_name(x) in usage_list:
                self.X.append(batch_x)

    def _parse_x_name(self, x):
        return x.split('/')[-1].split('.')[0]

    def _load_wav(self, wav_path):
        wav, sr = torchaudio.load(os.path.join(self.kaldi_root, wav_path))
        assert sr == self.sample_rate, f'Sample rate mismatch: real {sr}, config {self.sample_rate}'
        return wav.view(-1)

    def _load_transcript(self, table):
        """Load the transcripts for Librispeech"""
        def process_trans(transcript):
            #TODO: support character / bpe
            transcript = transcript.upper()
            return " ".join(list(transcript.replace(" ", "|"))) + " |"

        id_trsp_ls = []
        for meta_dir in table['meta_dir'].unique():
            with open(os.path.join(self.kaldi_root, meta_dir, "text")) as text_f:
                for line in text_f.readlines():
                    id, *transcript = line.strip().split()
                    transcript = ' '.join(transcript)
                    transcript = process_trans(transcript)

                    id_trsp_ls.append({'id': id, 'transcript': transcript})

        id_trsp_df = pd.DataFrame(id_trsp_ls)
        id_path_trsp_df = pd.merge(table, id_trsp_df, on='id')

        file_names = [self._parse_x_name(x) for x in id_path_trsp_df['file_path']]
        assert len(file_names) == len(set(file_names)), 'Duplicate file names found.'
        transcripts = id_path_trsp_df['transcript'].tolist()

        return dict(zip(file_names, transcripts))

    def _build_dictionary(self, transcripts, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        d = Dictionary()
        transcript_list = list(transcripts.values())
        Dictionary.add_transcripts_to_dictionary(
            transcript_list, d, workers
        )
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d


    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        wav_batch = [self._load_wav(x_file).numpy() for x_file in self.X[index]]
        label_batch = [self.Y[self._parse_x_name(x_file)].numpy() for x_file in self.X[index]]
        filename_batch = [Path(x_file).stem for x_file in self.X[index]]
        return wav_batch, label_batch, filename_batch # bucketing, return ((wavs, labels))

    def collate_fn(self, items):
        assert len(items) == 1
        return items[0][0], items[0][1], items[0][2] # hack bucketing, return (wavs, labels, filenames)
