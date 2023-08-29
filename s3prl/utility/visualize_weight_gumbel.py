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
parser.add_argument('-s', '--scale', type=float, default=1.0, required=False)
args = parser.parse_args()

assert os.path.isfile(args.ckpt), 'This has to be a ckpt file and not a directory.'
if len(args.name) == 0:
    args.name = args.ckpt.split('/')[-1] # use the ckpt name
if len(args.out_dir) == 0:
    args.out_dir = '/'.join(args.ckpt.split('/')[:-1]) # use the ckpt dir
else:
    os.mkdir(args.out_dir, exist_ok=True)

ckpt = torch.load(args.ckpt, map_location='cpu')
print('ckpt: ', ckpt.keys())
weights = ckpt['Featurizer']['weights'] * args.scale
temp = ckpt['Featurizer'].get('temp') or 1.0
print('Temperature: ', temp)
log_probs = F.log_softmax(weights/temp, dim=-1)
probs = log_probs.exp()
norm_weights = F.gumbel_softmax(log_probs, hard=True)
print('Normalized weights of upstream: \n', norm_weights)
weights = weights.cpu().double().tolist()
log_probs = log_probs.cpu().double().tolist()
probs = probs.cpu().double().tolist()
norm_weights = norm_weights.cpu().double().tolist()
print('Weights: \n', weights)
print('Probs: \n', probs)
print('Normalized weights: \n', norm_weights)

# plot weights
x = range(1, len(norm_weights)+1)
plt.bar(x, probs, align='center')

# set xticks and ylim
plt.xticks(x, [str(i-1) for i in x])
plt.ylim(0, 1)

# set names
plt.title(f'Distribution of normalized weight - {args.name}')
plt.xlabel('Layer ID (First -> Last)')
plt.ylabel('Weight')

plt.savefig(os.path.join(args.out_dir, 'visualize_weight.png'), bbox_inches='tight')