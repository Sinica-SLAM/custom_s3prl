# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ utils/visualize_weight.py ]
#   Synopsis     [ visualize the learned weighted sum from a downstream checkpoint ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import argparse
#-------------#
import torch
import torch.nn.functional as F
#-------------#
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, help='This has to be a ckpt not a directory.', required=True)
parser.add_argument('--name', type=str, default='', required=False)
parser.add_argument('--out_dir', type=str, default='', required=False)
args = parser.parse_args()

assert os.path.isfile(args.ckpt), 'This has to be a ckpt file and not a directory.'
if len(args.name) == 0:
    args.name = args.ckpt.split('/')[-2] # use the ckpt name
if len(args.out_dir) == 0:
    args.out_dir = '/'.join(args.ckpt.split('/')[:-1]) # use the ckpt dir
else:
    os.makedirs(args.out_dir, exist_ok=True)

ckpt = torch.load(args.ckpt, map_location='cpu')
print('Check point: ', list(ckpt.keys()))
featurizer = ckpt['Featurizer']
weights = featurizer['weights'].double() # (L, D)
if temp := featurizer.get('temp'):
    weights /= temp
    print(f'Temperature1: {temp.item()}')


# compute the layer selection counts
selects = weights.argmax(dim=0) # (D)
counts = torch.bincount(selects, minlength=weights.size(0)) # (L)
print('Layer selection counts: ')
for i, count in enumerate(counts):
    print(f"\tLayer {i:>2}: {count.item(): 3d}")

# compute the average variance of all dimension
probs = F.softmax(weights, dim=0)  # (L, D)
stds = probs.std(dim=0, correction=0) # (D)
stds = stds / F.one_hot(torch.tensor(0), num_classes=probs.size(0)).float().std(dim=0, correction=0)
print(f'Average anneal ratio: {stds.mean().item(): 0.4f}')
print(f'Max anneal ratio: {stds.max().item(): 0.4f} at dim {stds.argmax().item()}')

# plot weights
x = range(0, len(counts))
plt.bar(x, counts, color='b', alpha=0.7)

# set xticks and ylim
plt.xticks(x, x)
plt.ylim(0, weights.shape[1])  # set ylim

# set names
plt.title(f'Distribution of Layer select - {args.name}')
plt.xlabel('Layer ID (First -> Last)')
plt.ylabel('Count')

plt.savefig(os.path.join(args.out_dir, f'{args.name}_weight.png'), bbox_inches='tight')