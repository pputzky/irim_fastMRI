from torch.utils.data import DataLoader

from training_utils.data_transformers import TrainingTransform, TestingTransform
from training_utils.mri_data import SliceData
from fastMRI.common.subsample import RandomMaskFunc as MaskFunc


def create_training_datasets(args):
    train_mask = MaskFunc(args.center_fractions, args.accelerations)
    val_mask = MaskFunc(args.val_center_fractions, args.val_accelerations)

    train_data = SliceData(
        root=args.data_path / f'{args.challenge}_train',
        transform=TrainingTransform(train_mask, args.resolution, args.challenge, use_seed=False,
                                    train_resolution=args.train_resolution),
        sample_rate=args.sample_rate,
        challenge=args.challenge,
        n_slices=args.n_slices,
        use_rss=args.use_rss
    )

    val_data = SliceData(
        root=args.data_path / f'{args.challenge}_val',
        transform=TrainingTransform(val_mask, args.resolution, args.challenge, use_seed=True,
                                    train_resolution=None),
        sample_rate=1.,
        challenge=args.challenge,
        n_slices=-1
    )

    return val_data, train_data


def create_training_loaders(args):
    val_data, train_data = create_training_datasets(args)
    display_data = [val_data[i] for i in range(16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    display_loader = DataLoader(
        dataset=display_data,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, display_loader


def create_testing_loaders(args):
    mask_func = None
    if args.mask_kspace:
        mask_func = MaskFunc(args.center_fractions, args.accelerations)
    data = SliceData(
        root=args.data_path / f'{args.challenge}_{args.data_split}',
        transform=TestingTransform(args.challenge, mask_func),
        sample_rate=1.,
        challenge=args.challenge,
        n_slices=-1
    )
    data_loader = DataLoader(
        dataset=data,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return data_loader
