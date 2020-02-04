"""
This code is based on the training code found at
https://github.com/facebookresearch/fastMRI/blob/master/models/unet/train_unet.py
"""

import sys, os
import gc
import logging
import pathlib
import random
import shutil
import time

import numpy as np
import torch

import torchvision
from torch.utils.tensorboard import SummaryWriter

from irim.rim import RIM, ConvRNN
from irim import IRIM, InvertibleUnet, ResidualBlockPixelshuffle

from fastMRI.common import evaluate as numpy_eval
from fastMRI.common.args import Args
from fastMRI.data import transforms

from training_utils.models import IRIMfastMRI, RIMfastMRI
from training_utils.data_loaders import create_training_loaders
from training_utils.helpers import mse_gradient, estimate_to_image, image_loss, real_to_complex, complex_to_real

torch.cuda.current_device()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def train_epoch(args, epoch, model, train_loader, optimizer, writer):
    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(train_loader)

    memory_allocated = []
    for i, data in enumerate(train_loader):
        y, mask, target, metadata, target_norm, target_max = data
        y = y.to(args.device)
        mask = mask.to(args.device)
        target = target.to(args.device)
        target_norm = target_norm.to(args.device)
        target_max = target_max.to(args.device)

        optimizer.zero_grad()
        model.zero_grad()
        estimate = model.forward(y=y, mask=mask, metadata=metadata)
        if isinstance(estimate, list):
            loss = [image_loss(e, target, args, target_norm, target_max) for e in estimate]
            loss = sum(loss) / len(loss)
        else:
            loss = image_loss(estimate, target, args, target_norm, target_max)

        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if i > 0 else loss.item()
        writer.add_scalar('Loss', loss.item(), global_step + i)

        if args.device == 'cuda':
            memory_allocated.append(torch.cuda.max_memory_allocated() * 1e-6)
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.empty_cache()
        gc.collect()

        if i % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{i:4d}/{len(train_loader):4d}] '
                f'Loss = {loss.detach().item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s '
                f'Memory allocated (MB) = {np.min(memory_allocated):.2f}'
            )
            memory_allocated = []
        start_iter = time.perf_counter()
    optimizer.zero_grad()

    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer):
    model.eval()
    mse_losses = []
    psnr_losses = []
    nmse_losses = []
    ssim_losses = []
    memory_allocated = []

    start = time.perf_counter()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            y, mask, target, metadata = data[:4]
            y = y.to(args.device)
            mask = mask.to(args.device)
            target = target.to(args.device)
            if args.n_slices > 1:
                output = model.forward(y=y, mask=mask, metadata=metadata)
                output = estimate_to_image(output, args.resolution)
                output_np = output.to('cpu').transpose(0, -4).squeeze(-4)
                del output
            else:
                y = y.transpose(0, -4).squeeze(-4)
                mask = mask.squeeze(-4).repeat(y.size(0), 1, 1, 1, 1)
                metadata = metadata.repeat(y.size(0), 1)
                output_np = []
                for k, l in zip(range(0, y.size(0), args.batch_size),
                                range(args.batch_size, y.size(0) + args.batch_size, args.batch_size)):
                    output = model.forward(y=y[k:l], mask=mask[k:l], metadata=metadata[k:l])
                    output = estimate_to_image(output, args.resolution)
                    output_np.append(output.to('cpu'))
                output_np = torch.cat(output_np, 0)

            output_np = output_np.reshape(-1, output_np.size(-2), output_np.size(-1))
            target = target.reshape_as(output_np)

            output_np = output_np.to('cpu').numpy()
            target_np = target.to('cpu').numpy()
            mse_losses.append(numpy_eval.mse(target_np, output_np))
            psnr_losses.append(numpy_eval.psnr(target_np, output_np))
            nmse_losses.append(numpy_eval.nmse(target_np, output_np))
            ssim_losses.append(numpy_eval.ssim(target_np, output_np))

            if args.device == 'cuda':
                memory_allocated.append(torch.cuda.max_memory_allocated() * 1e-6)
                torch.cuda.reset_max_memory_allocated()

            del data, y, mask, target, metadata
            torch.cuda.empty_cache()

        writer.add_scalar('Val_MSE', np.mean(mse_losses), epoch)
        writer.add_scalar('Val_PSNR', np.mean(psnr_losses), epoch)
        writer.add_scalar('Val_NMSE', np.mean(nmse_losses), epoch)
        writer.add_scalar('Val_SSIM', np.mean(ssim_losses), epoch)
        writer.add_scalar('Val_memory', np.max(memory_allocated), epoch)

    return np.mean(nmse_losses), np.mean(psnr_losses), np.mean(mse_losses), np.mean(ssim_losses), \
           time.perf_counter() - start, np.max(memory_allocated)


def visualize(args, epoch, model, data_loader, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    output_images = []
    target_images = []
    corrupted_images = []
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            y, mask, target, metadata = data[:4]
            y = y.to(args.device)
            mask = mask.to(args.device)
            target = target.to(args.device)

            n_slices = target.size(-3)
            corrupted = transforms.root_sum_of_squares(estimate_to_image(
                y[..., n_slices // 2, :, :, :], target.size()[-2:]), 1)
            if args.n_slices == 1:
                mask = mask.squeeze(-4)
                y = y[..., n_slices // 2, :, :, :]

            estimate = model.forward(y=y, mask=mask, metadata=metadata)
            estimate.detach_()

            target = target[..., n_slices // 2, :, :]
            target_norm = target.norm(dim=(-2, -1), keepdim=True)
            corrupted_images.append(corrupted / target_norm)
            target_images.append(target / target_norm)
            if args.n_slices > 1:
                output_images.append(
                    estimate_to_image(estimate[..., n_slices // 2, :, :, :],
                                      target.size()[-2:]).clone().detach() / target_norm)
            else:
                output_images.append(estimate_to_image(estimate, target.size()[-2:]).clone().detach() / target_norm)

    output = torch.cat(output_images, 0)[:16].unsqueeze(1)
    target = torch.cat(target_images, 0)[:16].unsqueeze(1)
    corrupted = torch.cat(corrupted_images, 0)[:16].unsqueeze(1)

    save_image(target, 'Target')
    save_image(corrupted, 'Corrupted')
    save_image(output, 'Reconstruction')
    save_image(target - output, 'Error')
    save_image(corrupted - output, 'Corrupted_Reconstruction_Difference')
    save_image(corrupted - target, 'Corrupted_Target_Difference')


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def build_model(args):
    im_channels = 2 if args.challenge == 'singlecoil' else 30
    if args.use_rim:
        conv_nd = 3 if args.n_slices > 1 else 2
        rnn = ConvRNN(2 * im_channels, conv_dim=conv_nd)
        model = RIM(rnn, grad_fun=mse_gradient)
        model = RIMfastMRI(model, n_steps=args.n_steps)
    else:
        im_channels = im_channels * args.multiplicity
        channels = args.n_hidden if isinstance(args.n_hidden, list) else [args.n_hidden] * args.depth
        assert channels[0] >= 2 * im_channels + 8
        dilations = args.dilations
        n_hidden = args.n_network_hidden
        conv_nd = 3 if args.n_slices > 1 else 2
        if conv_nd == 3:
            # Make sure to not downsample in the slice direction
            dilations = [[1, d, d] for d in dilations]
        if args.shared_weights:
            cell_ = InvertibleUnet(n_channels=channels, n_hidden=n_hidden, dilations=dilations, conv_nd=conv_nd)
            cell = torch.nn.ModuleList([cell_] * args.n_steps)
        else:
            cell = torch.nn.ModuleList([InvertibleUnet(n_channels=channels, n_hidden=n_hidden,
                                                       dilations=dilations, conv_nd=conv_nd)
                                        for i in range(args.n_steps)])
        if args.parametric_output:
            output_function = ResidualBlockPixelshuffle(channels[0], 2, channels[0], conv_nd=conv_nd, use_glu=False)
        else:
            output_function = lambda x: complex_to_real(real_to_complex(x)[:,:im_channels // (2 * args.multiplicity)])

        model = IRIM(cell, grad_fun=mse_gradient, n_channels=im_channels)
        model = IRIMfastMRI(model, output_function, channels[0], multiplicity=args.multiplicity)

    return model.to(args.device)


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer


def build_optim(args, params):
    if args.optimizer.upper() == 'RMSPROP':
        optimizer = torch.optim.RMSprop(params, args.lr, weight_decay=args.weight_decay)
    if args.optimizer.upper() == 'ADAM':
        optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    if args.optimizer.upper() == 'SGD':
        optimizer = torch.optim.SGD(params, args.lr, weight_decay=args.weight_decay)
    return optimizer


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir / 'summary')

    checkpoint_pretrained = os.path.join(args.exp_dir, 'pretrained.pt')
    if args.checkpoint is None:
        checkpoint_path = os.path.join(args.exp_dir, 'model.pt')
    else:
        checkpoint_path = args.checkpoint

    if args.resume and os.path.exists(checkpoint_path):
        checkpoint, model, optimizer = load_model(checkpoint_path)
        args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch'] + 1
        del checkpoint
    else:
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model.parameters())
        if os.path.exists(checkpoint_pretrained):
            _, model, optimizer = load_model(checkpoint_pretrained)
            optimizer.lr = args.lr
        best_dev_loss = 1e9
        start_epoch = 0
    logging.info(args)
    logging.info(model)

    train_loader, val_loader, display_loader = create_training_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, writer)
        nmse_loss, psnr_loss, mse_loss, ssim_loss, dev_time, dev_mem = evaluate(args, epoch, model, val_loader, writer)
        visualize(args, epoch, model, display_loader, writer)
        scheduler.step(epoch)

        is_new_best = -ssim_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss, ssim_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'VAL_NMSE = {nmse_loss:.4g} VAL_MSE = {mse_loss:.4g} VAL_PSNR = {psnr_loss:.4g} '
            f'VAL_SSIM = {ssim_loss:.4g} \n'
            f'TrainTime = {train_time:.4f}s ValTime = {dev_time:.4f}s ValMemory = {dev_mem:.2f}',
        )
        if args.exit_after_checkpoint:
            writer.close()
            sys.exit(0)
    writer.close()


def create_arg_parser():
    parser = Args()
    # Mask parameters
    parser.add_argument('--val_accelerations', nargs='+', default=[8], type=int)
    parser.add_argument('--val_center_fractions', nargs='+', default=[0.04], type=float)
    parser.add_argument('--batch_size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--use_rim', action='store_true',
                        help='If set, RIM with fixed parameters')
    parser.add_argument('--optimizer', type=str, default='Adam', help="Optimizer to use choose between"
                                                                      "['Adam', 'SGD', 'RMSProp']")
    parser.add_argument('--loss', choices=['l1', 'mse', 'nmse', 'ssim'], default='ssim', help='Training loss')
    parser.add_argument('--loss_subsample', type=float, default=1., help='Sampling rate for loss mask')
    parser.add_argument('--use_rss', action='store_true',
                        help='If set, will train singlecoil model with RSS targets')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--n_slices', type=int, default=1, help='Number of slices in an observation. Default=1, if'
                                                                'n_slices > 1, we will use 3d convolutions')
    parser.add_argument('--lr_step_size', type=int, default=40, help='Period of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--n_steps', type=int, default=8, help='Number of RIM steps')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--report_interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--data_parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp_dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--multiplicity', type=int, default=1,
                        help='Number of eta estimates at every time step. The higher multiplicity, the lower the '
                             'number of necessary time steps would be expected.')
    parser.add_argument('--shared_weights', action='store_true',
                        help='If set, weights will be shared over time steps. (only relevant for IRIM)')
    parser.add_argument('--n_hidden', type=int, nargs='+', help='Number of hidden features in each layer. Can'
                                                                'be either Int or List of Ints')
    parser.add_argument('--n_network_hidden', type=int, nargs='+', help='Number of hidden features in each layer. Can'
                                                                        'be either Int or List of Ints')
    parser.add_argument('--dilations', type=int, nargs='+', help='Kernel dilations in each in each layer. Can'
                                                                 'be either Int or List of Ints')
    parser.add_argument('--depth', type=int, help='Number of RNN layers.')
    parser.add_argument('--train_resolution', type=int, nargs=2, default=None, help='Image resolution during training')
    parser.add_argument('--parametric_output', action='store_true', help='Use a parametric function for map the'
                                                                         'last layer of the iRIM to an image estimate')
    parser.add_argument('--exit_after_checkpoint', action='store_true')

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
