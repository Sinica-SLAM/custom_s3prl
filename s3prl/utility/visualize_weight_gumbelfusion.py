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
print('ckpt: ', list(ckpt.keys()))
weights = ckpt['Featurizer']['weights'].double() # (L, D)
temp = ckpt['Featurizer'].get('temp') or 1.0
print(f"Temperature: {temp.item() :0.6f}")

# compute the layer selection counts
selects = weights.argmax(dim=0) # (D)
counts = torch.bincount(selects, minlength=weights.size(0)) # (L)
print('Layer selection counts: ')
for i, count in enumerate(counts):
    print(f"Layer {i}: {count.item(): 3d}")

# compute the average variance of all dimension
anneal_weights = F.softmax(weights/temp, dim=0) # (L, D), which is the probability distribution against layer of each dimension
layers = torch.arange(0, weights.size(0)).double().unsqueeze(-1) # (L, 1)
square_means = (anneal_weights * layers.pow(2)).sum(dim=0) # (D)
mean_squares = (anneal_weights * layers).sum(dim=0).pow(2) # (D)
standard_deviations = (square_means - mean_squares + 1e-10).sqrt() # (D)
print(f'Average standard deviations: {standard_deviations.mean().item(): 0.4f}')
print(f'Max standard deviations: {standard_deviations.max().item(): 0.4f} at dim {standard_deviations.argmax().item()}')

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