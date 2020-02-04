import torch

from irim import IRIM, MemoryFreeInvertibleModule
from irim.rim import RIM

from training_utils.helpers import complex_to_real, real_to_complex


class RescaleByStd(object):
    def __init__(self, slack=1e-6):
        self.slack = slack

    def forward(self, data, gamma=None):
        if gamma is None:
            gamma = data.std(dim=list(range(1,data.dim())), keepdim=True) + self.slack
        data = data / gamma
        return data, gamma

    def reverse(self, data, gamma):
        data = data * gamma
        return data


class RIMfastMRI(torch.nn.Module):
    def __init__(self, model, preprocessor=RescaleByStd(), n_steps=8):
        """
        An RIM model wrapper for the fastMRI challenge.
        :param model: RIM model
        :param preprocessor: a function that rescales each sample
        :param n_steps: Number of RIM steps [int]
        """
        super().__init__()
        assert isinstance(model, RIM)
        self.model = model
        self.preprocessor = preprocessor
        self.n_steps = n_steps

    def forward(self, y, mask, metadata=None):
        """
        :param y: Zero-filled kspace reconstruction [Tensor]
        :param mask: Sub-sampling mask
        :param metadata: will be ignored
        :return: complex valued image estimate
        """
        accumulate_eta = self.training
        y, gamma = self.preprocessor.forward(y)

        eta = complex_to_real(y)
        eta, hx = self.model.forward(eta, [y, mask], n_steps=self.n_steps, accumulate_eta=accumulate_eta)

        if accumulate_eta:
            eta = [real_to_complex(e) for e in eta]
            eta = [self.preprocessor.reverse(e, gamma) for e in eta]
        else:
            eta = real_to_complex(eta)
            eta = self.preprocessor.reverse(eta, gamma)

        return eta


class IRIMfastMRI(torch.nn.Module):
    def __init__(self, model, output_function, n_latent, preprocessor=RescaleByStd(), multiplicity=1):
        """
        An i-RIM wrapper for the fastMRI data
        :param model: i-RIM model
        :param output_function: function that maps the output if the i-RIM to image space
        :param n_latent: number of channels in the machine state
        :param preprocessor: a function that rescales each sample
        :param multiplicity: number of virtual samples at each time step
        """
        super().__init__()
        assert isinstance(model, IRIM)
        self.model = MemoryFreeInvertibleModule(model)
        self.output_function = output_function
        self.n_latent = n_latent
        self.preprocessor = preprocessor
        self.multiplicity = multiplicity

    def forward(self, y, mask, metadata=None):
        """
        :param y: Zero-filled kspace reconstruction [Tensor]
        :param mask: Sub-sampling mask
        :param metadata: Tensor with metadata
        :return: complex valued image estimate
        """
        y, gamma = self.preprocessor.forward(y)
        y = torch.cat(self.multiplicity*[y],1)
        eta = complex_to_real(y)
        x = torch.cat((eta, eta.new_zeros((eta.size(0), self.n_latent - eta.size(1)) + eta.size()[2:])), 1)

        if metadata is not None:
            while len(metadata.size()) < len(eta.size()):
                metadata = metadata.unsqueeze(-1)

            x[:, -metadata.size(1):] = metadata

        x = self.model.forward(x, [y,mask])
        eta = self.output_function(x)
        eta = real_to_complex(eta)
        eta = self.preprocessor.reverse(eta, gamma)

        return eta
