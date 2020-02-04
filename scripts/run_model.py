"""
This code is based on code found at
https://github.com/facebookresearch/fastMRI/blob/master/models/unet/run_unet.py
"""

import pathlib
import sys
from collections import defaultdict

import numpy as np
import torch

from fastMRI.common.args import Args
from fastMRI.common.utils import save_reconstructions
from .train_model import load_model, estimate_to_image

import gc

from training_utils.data_loaders import create_testing_loaders

torch.backends.cudnn.benchmark = False


def run_rim(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for i, (y, mask, metadata, fnames, slices) in enumerate(data_loader):
            print('Reconstruction '+str(i))

            y = y.to(args.device)
            mask = mask.to(args.device)
            if args.n_slices > 1:
                estimate = model.forward(y=y, mask=mask, metadata=metadata)
                estimate = estimate_to_image(estimate, args.resolution)
                output = estimate.to('cpu').transpose(0, -4).squeeze(-4)
                del estimate
            else:
                y = y.transpose(0, -4).squeeze(-4)
                mask = mask.squeeze(-4).repeat(y.size(0), 1, 1, 1, 1)
                metadata = metadata.repeat(y.size(0), 1)
                output = []
                for k, l in zip(range(0, y.size(0), args.batch_size),
                                range(args.batch_size, y.size(0) + args.batch_size, args.batch_size)):
                    estimate = model.forward(y=y[k:l], mask=mask[k:l], metadata=metadata[k:l])
                    estimate = estimate_to_image(estimate, args.resolution)
                    output.append(estimate.to('cpu'))
                output = torch.cat(output, 0)

            for i in range(output.shape[0]):
                reconstructions[fnames[0]].append((slices[i].numpy(), output[i].numpy()))

            gc.collect()
            torch.cuda.empty_cache()

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }
    return reconstructions

def main(args):
    data_loader = create_testing_loaders(args)
    checkpoint, model, optimizer = load_model(args.checkpoint)
    args.n_slices = checkpoint['args'].n_slices
    print('Reconstructing...')
    reconstructions = run_rim(args, model, data_loader)
    print('Saving...')
    save_reconstructions(reconstructions, args.out_dir)
    print('Done!')


def create_arg_parser():
    parser = Args()
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--mask-kspace', action='store_true',
                        help='Whether to apply a mask (set to True for val data and False '
                             'for test data')
    parser.add_argument('--data-split', choices=['val', 'test', 'test_v2', 'challenge'], required=True,
                        help='Which data partition to run on: "val" or "test"')
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the RIM model')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    parser.add_argument('--data_parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
