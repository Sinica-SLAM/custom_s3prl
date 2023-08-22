###############
# IMPORTATION #
###############
# wget --show-progress https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz
import os
import argparse
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
import pandas as pd


#############################
# PREPROCESS CONFIGURATIONS #
#############################
def get_preprocess_args():

    parser = argparse.ArgumentParser(description='preprocess arguments for any dataset.')

    parser.add_argument('-i', '--input_data', default='./data', type=str, help='Path to your LibriLight directory', required=False)
    parser.add_argument('-o', '--output_path', default='./data', type=str, help='Path to store output', required=False)
    parser.add_argument('-n', '--name', default='len_for_bucket', type=str, help='Name of the output directory', required=False)
    parser.add_argument('--n_jobs', default=-1, type=int, help='Number of jobs used for feature extraction', required=False)

    args = parser.parse_args()
    return args


##################
# EXTRACT LENGTH #
##################
def extract_length(input_file):
    torchaudio.set_audio_backend("sox_io")
    return torchaudio.info(input_file).num_frames


###################
# GENERATE LENGTH #
###################
def generate_length(args, tr_set):

    root_dir = os.path.realpath(args.input_data)

    assert os.path.exists(root_dir), \
             f'Please download LibriLight dataset and put it in {args.input_data}'

    for i, s in enumerate(tr_set):
        match s:
            case 'hakka1-train':
                meta_dir = 'data/train'
            case 'hakka1-dev':
                meta_dir = 'data/dev'
            case 'hakka1-test':
                meta_dir = 'data/test'
            case 'hakka2-train':
                meta_dir = 'data2/train'
            case 'hakka2-dev':
                meta_dir = 'data2/dev'
            case 'hakka2-test':
                meta_dir = 'data2/test'
            case _:
                raise NotImplementedError

        wavscp = os.path.join(root_dir, meta_dir, 'wav.scp')
        table = pd.read_csv(wavscp, delim_whitespace=True, header=None, names=['id', 'file_path'])

        table.insert(1, 'meta_dir', meta_dir)
        table['file_path'] = table.apply(lambda row: os.path.relpath(os.path.realpath(row['file_path']), root_dir), axis=1)

        print('')
        print(f'Preprocessing data in: {s}, {len(table)} audio files found.')

        output_dir = os.path.join(args.output_path, args.name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print('Extracting audio length...', flush=True)
        table['length'] = Parallel(n_jobs=args.n_jobs)(delayed(extract_length)(os.path.join(root_dir, str(file))) for file in tqdm(table['file_path']))
        table['label'] = None

        # sort by len
        table.sort_values(by=['length'], ascending=False, inplace=True, ignore_index=True)

        # Dump data
        lengths = table['length'].tolist()
        print('Total:{}mins, Min:{}secs, Max:{}secs'.format(sum(lengths)//960000, min(lengths or [0])//16000, max(lengths or [0])//16000))
        table.to_csv(os.path.join(output_dir, tr_set[i] + '.csv'), index=False)

    print('All done, saved at', output_dir, 'exit.')


########
# MAIN #
########
def main():

    # get arguments
    args = get_preprocess_args()

    SETS = ['hakka1-train', 'hakka1-dev', 'hakka1-test', 'hakka2-train', 'hakka2-dev', 'hakka2-test']

    # Select data sets
    for idx, s in enumerate(SETS):
        print('\t', idx, ':', s)
    tr_set = input('Please enter the index of splits you wish to use preprocess. (seperate with space): ')
    tr_set = [SETS[int(t)] for t in tr_set.split(' ')]

    # Acoustic Feature Extraction & Make Data Table
    generate_length(args, tr_set)


if __name__ == '__main__':
    main()
