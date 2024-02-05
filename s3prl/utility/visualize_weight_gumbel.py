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
weights = featurizer['weights'].double() # (L)
origin_probs = F.softmax(weights, dim=-1)
if temp := featurizer.get('temp'):
    weights /= temp
    print(f'Temperature1: {temp.item()}')
probs = F.softmax(weights, dim=-1)
std = probs.std(dim=0, correction=0)
std = std / F.one_hot(probs.argmax(), num_classes=probs.size(0)).float().std(dim=0, correction=0)

origin_probs = origin_probs.cpu().tolist()
probs = probs.cpu().tolist()
print('Probs:')
for i, p in enumerate(probs):
    print(f'\tLayer {i:>2}: {p}')
print(f'Anneal ratio: {std.cpu().item():0.4f}')

# plot weights
x = range(0, len(probs))
# plot original probs in blue, with highiest prob in red
plt.bar(x, origin_probs, color=['b' if i != max(origin_probs) else 'r' for i in origin_probs], alpha=0.7)
# plot annealed probs in red overlapped with original probs
#plt.bar(x, probs, color='r', alpha=0.7)

# set xticks and ylim
plt.xticks(x, x)
plt.ylim(0)

# set names
plt.title(f'Distribution of normalized weight - {args.name}')
plt.xlabel('Layer ID (First -> Last)')
plt.ylabel('Weight')

plt.savefig(os.path.join(args.out_dir, f'{args.name}_weight.png'), bbox_inches='tight')