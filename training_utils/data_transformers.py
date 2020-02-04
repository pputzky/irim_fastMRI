from fastMRI.data import transforms
import numpy as np
import torch


class TrainingTransform:
    """
    Data Transformer for training models.
    """

    def __init__(self, mask_func, resolution, which_challenge, train_resolution=None, use_seed=True):
        """
        :param mask_func: common.subsample.MaskFunc, A mask sampling function
        :param resolution: [Int] Resolution of target
        :param which_challenge: Either "singlecoil" or "multicoil" denoting the dataset.
        :param train_resolution: Resolution of k-space measurement. Cropping is done in image space
                                 before masking
        :param use_seed: If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.train_resolution = train_resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, target, attrs, fname, slice, n_slices=1):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int/list): Serial number(s) of the slice(s). Will be a list for volumes and an int for slices.
            n_slice (int): Number of slices to output.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                mask (torch.Tesnor): k-space sampling mask
                target (torch.Tensor): Target image converted to a torch Tensor.
                metadata(torch.Tensor): 1-hot vector indicating measurement setup
        """
        seed = None if not self.use_seed else tuple(map(ord, fname))
        np.random.seed(seed)

        kspace = transforms.to_tensor(kspace)
        target = transforms.to_tensor(target)
        target = transforms.center_crop(target, (self.resolution, self.resolution))

        if n_slices > 1:
            total_slices = kspace.size(0)
            slice_id = np.random.randint(0, max(total_slices - n_slices, 1))
            kspace = kspace[slice_id:slice_id + n_slices]
            target = target[slice_id:slice_id + n_slices]
        if n_slices < 0:
            n_slices = kspace.shape[0]

        if self.train_resolution is not None:
            kspace = transforms.ifft2(kspace)
            p = max(kspace.size(-3) - self.train_resolution[0], kspace.size(-2) - self.train_resolution[1]) // 2 + 1
            kspace = torch.nn.functional.pad(input=kspace, pad=(0, 0, p, p, p, p), mode='constant', value=0)
            kspace = transforms.complex_center_crop(kspace, self.train_resolution)
            kspace = transforms.fft2(kspace)

        # Apply mask
        if self.mask_func is not None:
            masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace
            mask = kspace != 0
            mask = mask.to(kspace)[..., :1, :, :1]
            mask = mask[:1, ...]

        if self.which_challenge == 'multicoil':
            if masked_kspace.dim() == 5:
                masked_kspace = masked_kspace.transpose(0, 1)
                mask = mask.transpose(0, 1)
        else:
            masked_kspace = masked_kspace.unsqueeze(0)
            mask = mask.unsqueeze(0)

        data_norm = attrs['norm'].astype(np.float32) * n_slices**0.5
        return transforms.ifft2(masked_kspace), mask, target, \
               transforms.to_tensor(np.array(attrs['metadata'], np.float32)), \
               data_norm, attrs['max'].astype(np.float32)


class TestingTransform:
    """
    Data Transformer for running models on a test dataset.
    """

    def __init__(self, which_challenge, mask_func=None):
        """
        Args:
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.which_challenge = which_challenge
        self.mask_func = mask_func

    def __call__(self, kspace, target, attrs, fname, slice, n_slices=-1):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int/list): Serial number(s) of the slice(s). Will be a list for volumes and an int for slices.
            n_slice (int): Number of slices to output.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                mask (torch.Tesnor): k-space sampling mask
                metadata(torch.Tensor): 1-hot vector indicating measurement setup
                fname (pathlib.Path): Path to the input file
                slice (int): Serial number of the slice
        """
        kspace = transforms.to_tensor(kspace)

        if self.mask_func is not None:
            seed = tuple(map(ord, fname))
            masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace
            mask = kspace != 0
            mask = mask.to(kspace)[..., :1, :, :1]
            mask = mask[:1, ...]

        if self.which_challenge == 'multicoil':
            if masked_kspace.dim() == 5:
                masked_kspace = masked_kspace.transpose(0, 1)
                mask = mask.transpose(0, 1)
        else:
            masked_kspace = masked_kspace.unsqueeze(0)
            mask = mask.unsqueeze(0)

        return transforms.ifft2(masked_kspace), mask, \
               transforms.to_tensor(np.array(attrs['metadata'], np.float32)), \
               fname, slice
