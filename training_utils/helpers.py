import torch
from torch.nn import functional as F

from fastMRI.data import transforms
from training_utils import ssim


def real_to_complex(x):
    return torch.stack(torch.chunk(x,2,1),-1)


def complex_to_real(x):
    return torch.cat((x[...,0],x[...,1]),1)


def mse_gradient(x, data):
    """
    Calculates the gradient under a linear forward model
    :param x: image estimate
    :param data: [y,mask], y - zero-filled image reconstruction, mask - sub-sampling mask
    :return: image gradient
    """

    y, mask = data[0], data[1]
    x = real_to_complex(x)
    x = transforms.fft2(x)
    x = mask * x
    x = transforms.ifft2(x)
    x = x - y

    x = complex_to_real(x)
    return x


def estimate_to_image(estimate, resolution=None):
    if resolution is not None:
        if isinstance(resolution, int):
            resolution = (resolution,resolution)
        estimate = transforms.complex_center_crop(estimate, tuple(resolution))
    image = transforms.complex_abs(estimate)
    # If we are dealing with a multi-channel image, produce RSS image
    image = transforms.root_sum_of_squares(image, 1)

    return image


def image_loss(estimate, target, args, target_norm=None, target_max=None):
    loss_selector = {'l1': lambda x, t: F.l1_loss(x, t, reduction='none'),
                     'mse': lambda x, t: F.mse_loss(x, t, reduction='none'),
                     'nmse': lambda x, t: F.mse_loss(x, t, reduction='none'),
                     'ssim': lambda x, t: -ssim.ssim_uniform(x, t, window_size=7, reduction='none'),
                     }
    loss_fun = loss_selector[args.loss]

    image = estimate_to_image(estimate, target.size()[-2:])
    image = image.reshape(-1, 1, image.size(-2), image.size(-1))
    target = target.reshape_as(image)
    if args.loss == 'ssim':
        normalizer = target_max
        for i in range(len(target.size()) - 1):
            normalizer = normalizer.unsqueeze(-1)
        target = target / normalizer
        image = image / normalizer
    if args.loss == 'nmse':
        normalizer = target_norm
        for i in range(len(target.size()) - 1):
            normalizer = normalizer.unsqueeze(-1)
        target = target / normalizer
        image = image / normalizer

    mask = torch.ones_like(target)
    if 0. < args.loss_subsample < 1.:
        mask = torch.torch.bernoulli(args.loss_subsample * mask)

    loss = mask * loss_fun(image,target)
    loss = loss.sum((-3,-2,-1)) / mask.sum((-3,-2,-1))
    if args.loss == 'nmse':
        loss = loss * mask[0].numel()

    return loss.mean()
