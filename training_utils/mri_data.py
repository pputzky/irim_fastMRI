import pathlib
import random

import h5py
from torch.utils.data import Dataset
import xmltodict


class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image volumes.
    """

    def __init__(self, root, transform, challenge, sample_rate=1, n_slices=1, use_rss=False):
        """

        :param root: Path to the dataset
        :param transform: A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
        :param challenge: "singlecoil" or "multicoil" depending on which challenge to use.
        :param sample_rate: A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        :param n_slices: Number of slices in a volume. default: 1
        :param use_rss: If True, uses RSS images as targets, also in the singlecoil case
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' and not use_rss \
            else 'reconstruction_rss'

        self.examples = []
        self.n_slices = min(n_slices,1)
        files = list(pathlib.Path(root).iterdir())
        files = [_ for _ in files if _.name.endswith('.h5')]
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):
            kspace = h5py.File(fname, 'r')['kspace']
            num_slices = kspace.shape[0]
            if n_slices == 1:
                self.examples += [(fname, slice) for slice in range(num_slices)]
            elif n_slices == 0:
                self.examples += [(fname, 0)]
            else:
                self.examples += [(fname, range(num_slices))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice].squeeze()
            target = data[self.recons_key][slice].squeeze() if self.recons_key in data else None
            contrast_type, metadata = export_attrs(data['ismrmrd_header'], data.attrs['acquisition'])
            attributes = dict(data.attrs.items())
            attributes['contrast_type'] = contrast_type
            attributes['metadata'] = metadata
            if 'norm' in attributes:
                attributes['norm'] = attributes['norm'] / data['kspace'].shape[0]**0.5
            return self.transform(kspace, target, attributes, fname.name, slice, n_slices=self.n_slices)

def export_attrs(ismrmrd_header, acquisition):
    xml_header = ismrmrd_header[()].decode('UTF-8')
    dict_header = xmltodict.parse(xml_header)
    usefull_info = ['studyInformation', 'measurementInformation', 'acquisitionSystemInformation',
                    'experimentalConditions', 'encoding', 'sequenceParameters', 'userParameters']
    ismrmrd_header_to_dict = {}
    for keys, values in dict_header.items():
        for key, value in values.items():
            if key in usefull_info:
                for k, v in value.items():
                    if key != 'encoding':
                        if (key == 'acquisitionSystemInformation' and k == 'coilLabel'
                        ) or (key == 'userParameters' and k == 'userParameterDouble'):
                            for cnk, cnv in v[0].items():
                                ismrmrd_header_to_dict[key + '_' + k + '_' + cnk] = cnv
                        else:
                            ismrmrd_header_to_dict[key + '_' + k] = v
                    else:
                        if key == 'encoding' and k == 'trajectory':
                            ismrmrd_header_to_dict[k] = v
                        elif key == 'encoding' and k == 'parallelImaging':
                            for enc_pi_key, enc_pi_value in v.items():
                                if enc_pi_key == 'calibrationMode':
                                    ismrmrd_header_to_dict[k + '_' + enc_pi_key] = enc_pi_value
                                else:
                                    for enc_pi_k, enc_pi_v in enc_pi_value.items():
                                        ismrmrd_header_to_dict[k + '_' + enc_pi_key + '_' + enc_pi_k] = enc_pi_v
                        else:
                            for enc_key, enc_value in v.items():
                                for enc_k, enc_v in enc_value.items():
                                    ismrmrd_header_to_dict[k + '_' + enc_key + '_' + enc_k] = enc_v

    if acquisition == 'CORPDFS_FBKREPEAT':
        acquisition = 'CORPDFS_FBK'

    if acquisition == 'CORPDFS_FBK' and ismrmrd_header_to_dict[
        'acquisitionSystemInformation_systemModel'] == 'Aera':
        flag = 'fs_1_5T_Aera'
        features = [1, 0, 0, 0, 0, 0, 0, 0]
    elif acquisition == 'CORPD_FBK' and ismrmrd_header_to_dict[
        'acquisitionSystemInformation_systemModel'] == 'Aera':
        flag = 'non_fs_1_5T_Aera'
        features = [0, 1, 0, 0, 0, 0, 0, 0]
    elif acquisition == 'CORPDFS_FBK' and ismrmrd_header_to_dict[
        'acquisitionSystemInformation_systemModel'] == 'Biograph_mMR':
        flag = 'fs_3T_Biograph'
        features = [0, 0, 1, 0, 0, 0, 0, 0]
    elif acquisition == 'CORPD_FBK' and ismrmrd_header_to_dict[
        'acquisitionSystemInformation_systemModel'] == 'Biograph_mMR':
        flag = 'non_fs_3T_Biograph'
        features = [0, 0, 0, 1, 0, 0, 0, 0]
    elif acquisition == 'CORPDFS_FBK' and ismrmrd_header_to_dict[
        'acquisitionSystemInformation_systemModel'] == 'Prisma_fit':
        flag = 'fs_3T_Prisma'
        features = [0, 0, 0, 0, 1, 0, 0, 0]
    elif acquisition == 'CORPD_FBK' and ismrmrd_header_to_dict[
        'acquisitionSystemInformation_systemModel'] == 'Prisma_fit':
        flag = 'non_fs_3T_Prisma'
        features = [0, 0, 0, 0, 0, 1, 0, 0]
    elif acquisition == 'CORPDFS_FBK' and ismrmrd_header_to_dict[
        'acquisitionSystemInformation_systemModel'] == 'Skyra':
        flag = 'fs_3T_Skyra'
        features = [0, 0, 0, 0, 0, 0, 1, 0]
    elif acquisition == 'CORPD_FBK' and ismrmrd_header_to_dict[
        'acquisitionSystemInformation_systemModel'] == 'Skyra':
        flag = 'non_fs_3T_Skyra'
        features = [0, 0, 0, 0, 0, 0, 0, 1]

    return flag, features
