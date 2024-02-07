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

from s3prl.utility.visualize_weight_gumbel import visualize_weight_gumbel
from s3prl.utility.visualize_weight_gumbelfusion import visualize_weight_gumbelfusion

def visualize_multiple_weight(ckpt, F1, F2):
    if 'gumbelfusion' in F1.lower():
        prob1 = visualize_weight_gumbelfusion(ckpt['iFeaturizer1'])
    else:
        _, prob1 = visualize_weight_gumbel(ckpt['ifeaturizer1'])
    print('')
    if 'gumbelfusion' in F2.lower():
        prob2 = visualize_weight_gumbelfusion(ckpt['iFeaturizer2'])
    else:
        _, prob2 = visualize_weight_gumbel(ckpt['ifeaturizer2'])

    if fusioner := ckpt.get('Fusioner'):
        if lamb := fusioner.get('lamb'):
            true_lamb = torch.sigmoid(lamb).item()
            print('Lambda: ', true_lamb)
            if prob1 is not None:
                prob1 *= true_lamb
            if prob2 is not None:
                prob2 *= (1-true_lamb)
        gate_values = fusioner.get('gate_values')
        if gate_values is not None:
            if temp := fusioner.get('temp'):
                #gate_values /= temp
                print(f'Temperature: {temp.item()}')
            gates = torch.sigmoid(gate_values).tolist()
            gates = list(enumerate(gates))
            gates.sort(key=lambda x: x[1])
            print('Gates: ')
            gate1 = 0
            gate2 = 0
            for i, gate in gates:
                print(f'Dim {i}: {gate:.4f}')
                if gate > 0.5:
                    gate1 += 1
                else:
                    gate2 += 1
            print(f'Gate1: {gate1}, Gate2: {gate2}')

    return prob1, prob2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        help='This has to be a ckpt not a directory.', required=True)
    parser.add_argument('--name', type=str, default='', required=False)
    parser.add_argument('--out_dir', type=str, default='', required=False)
    parser.add_argument('-u1', '--upstream1', help='upstream1 in run_downstream', type=str, required=True)
    parser.add_argument('-u2', '--upstream2', help='upstream2 in run_downstream', type=str, required=True)
    parser.add_argument('-F1', help="The the first upstream's featurizer", default="", type=str)
    parser.add_argument('-F2', help="The the second upstream's featurizer", default="", type=str)
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
    print(f"Step: {ckpt['Step']}")
    prob1, prob2 = visualize_multiple_weight(ckpt, args.F1, args.F2)

    # plot weights
    if prob1 or prob2:
        upstream1 = args.upstream1
        upstream2 = args.upstream2
        x = range(0, len(prob1 or prob2))
        if prob1 is not None:
            plt.bar(x, prob1, 0.3, align='edge', color='deepskyblue')
        if prob2 is not None:
            plt.bar(x, prob2, -0.3, align='edge', color='orange')
        # set xticks and ylim
        plt.xticks(x, x)
        plt.ylim(0, max(sum(prob1) if prob1 else 1, sum(prob2) if prob2 else 1))
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