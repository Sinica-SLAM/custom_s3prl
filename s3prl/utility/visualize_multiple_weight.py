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
# -------------#
# -------------#
import matplotlib.pyplot as plt
import os
import argparse
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str,
                    help='This has to be a ckpt not a directory.', required=True)
parser.add_argument('--name', type=str, default='', required=False)
parser.add_argument('--out_dir', type=str, default='', required=False)
parser.add_argument('-u1', '--upstream1', help='upstream1 in run_downstream', type=str, required=True)
parser.add_argument('-u2', '--upstream2', help='upstream2 in run_downstream', type=str, required=True)
args = parser.parse_args()

assert os.path.isfile(
    args.ckpt), 'This has to be a ckpt file and not a directory.'
if len(args.name) == 0:
    args.name = args.ckpt.split('/')[-2]  # use the ckpt dir name
if len(args.out_dir) == 0:
    args.out_dir = '/'.join(args.ckpt.split('/')[:-1])  # use the ckpt dir
else:
    os.makedirs(args.out_dir, exist_ok=True)

ckpt = torch.load(args.ckpt, map_location='cpu')
print('Check point: ', list(ckpt.keys()))

norm_weights1 = None
norm_weights2 = None
if ifeaturizer1 := ckpt.get('iFeaturizer1'):
    weights1 = ifeaturizer1['weights'].double()
    if temp1 := ifeaturizer1.get('temp'):
        weights1 /= temp1
        print(f'Temperature1: {temp1.item():.4f}')
    norm_weights1 = F.softmax(weights1, dim=-1)
    print('Normalized weights of upstream1: \n', norm_weights1)
if ifeaturizer2 := ckpt.get('iFeaturizer2'):
    weights2 = ifeaturizer2['weights'].double()
    if temp2 := ifeaturizer2.get('temp'):
        weights2 /= temp2
        print(f'Temperature2: {temp2.item():.4f}')
    norm_weights2 = F.softmax(weights2, dim=-1)
    print('Normalized weights of upstream2: \n', norm_weights2)

if fusioner := ckpt.get('Fusioner'):
    if lamb := fusioner.get('lamb'):
        true_lamb = torch.sigmoid(lamb).item()
        print('Lambda: ', true_lamb)
        if norm_weights1 is not None:
            norm_weights1 *= true_lamb
        if norm_weights2 is not None:
            norm_weights2 *= (1-true_lamb)
norm_weights1 = norm_weights1.cpu().tolist() if norm_weights1 is not None else None
norm_weights2 = norm_weights2.cpu().tolist() if norm_weights2 is not None else None


upstream1 = args.upstream1
upstream2 = args.upstream2
# plot weights
x = range(0, len(norm_weights1 or norm_weights2))
if norm_weights1 is not None:
    plt.bar(x, norm_weights1, 0.3, align='edge', color='deepskyblue')
if norm_weights2 is not None:
    plt.bar(x, norm_weights2, -0.3, align='edge', color='orange')
# set xticks and ylim
plt.xticks(x, x)
# plt.ylim(0, 0.4)
# set names
plt.title(f'Distribution of normalized weight - {args.name}')
plt.xlabel('Layer ID (First -> Last)')
plt.ylabel('Weight')
# set legend
colors = {upstream1: 'deepskyblue', upstream2: 'orange'}
labels = list(colors.keys())
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label])
           for label in labels]
plt.legend(handles, labels)
img_path = os.path.join(args.out_dir, f'{args.name}_weight.png')
plt.savefig(img_path, bbox_inches='tight')
print("Image saved at " + img_path )
