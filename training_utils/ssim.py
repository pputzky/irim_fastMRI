import torch
import torch.nn.functional as F


def get_uniform_window(window_size, n_channels):
    window = torch.ones(n_channels, 1, window_size, window_size, requires_grad=False)
    window = window / (window_size ** 2)
    return window


def reflection_pad(x, window_size):
    pad_width = window_size // 2
    x = F.pad(x, [pad_width, pad_width, pad_width, pad_width], mode='reflect')

    return x


def conv2d_with_reflection_pad(x, window):
    x = reflection_pad(x, window_size=window.size(-1))
    x = F.conv2d(x, window, padding=0, groups=x.size(1))

    return x


def calc_ssim(x1, x2, window, C1=0.01, C2=0.03):
    """
    This function calculates the pixel-wise SSIM in a window-sized area, under the assumption
    that x1 and x2 have pixel values in range [0,1]. The default values for C1 and C2 are chosen
    in accordance with the scikit-image default values

    :param x1: 2d image
    :param x2: 2d image
    :param window: 2d convolution kernel
    :param C1: positive scalar, luminance fudge parameter
    :param C2: positive scalar, contrast fudge parameter
    :return: pixel-wise SSIM
    """
    x = torch.cat((x1, x2), 0)
    mu = conv2d_with_reflection_pad(x, window)
    mu_squared = mu ** 2
    mu_cross = mu[:x1.size(0)] * mu[x1.size(0):]

    var = conv2d_with_reflection_pad(x * x, window) - mu_squared
    var_cross = conv2d_with_reflection_pad(x1 * x2, window) - mu_cross

    luminance = (2 * mu_cross + C1 ** 2) / (mu_squared[:x1.size(0)] + mu_squared[x1.size(0):] + C1 ** 2)
    contrast = (2 * var_cross + C2 ** 2) / (var[:x1.size(0)] + var[x1.size(0):] + C2 ** 2)
    ssim_val = luminance * contrast
    ssim_val = ssim_val.mean(1, keepdim=True)

    return ssim_val

def ssim_uniform(input, target, window_size=11, reduction='mean'):
    """
    Calculates SSIM using a uniform window. This approximates the scikit-image implementation used
    in the fastMRI challenge. This function assumes that input and target are in range [0,1]
    :param input: 2D image tensor
    :param target: 2D image tensor
    :param window_size: integer
    :param reduction: 'mean', 'sum', or 'none', see pytorch reductions
    :return: ssim value
    """
    window = get_uniform_window(window_size, input.size(1))
    window = window.to(input.device)
    ssim_val = calc_ssim(input, target, window)
    if reduction == 'mean':
        ssim_val = ssim_val.mean()
    elif not (reduction is None or reduction == 'none'):
        ssim_val = ssim_val.sum()

    return ssim_val
