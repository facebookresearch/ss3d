# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import absolute_import, division, print_function

import math

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


def normalize_imagenet(x):
    """Normalize input images according to ImageNet standards.

    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x


class SineFiLMLayer(nn.Module):
    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30.0
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, x, frequencies=1.0, shift=0.0):
        return torch.sin(frequencies * self.omega_0 * self.linear(x) + shift)

    def forward_with_intermediate(self, x, frequencies=1.0, shift=0.0):
        # For visualization of activation distributions
        intermediate = frequencies * self.omega_0 * self.linear(x) + shift
        return torch.sin(intermediate), intermediate


class SirenFiLM(nn.Module):
    def __init__(
        self,
        dim=3,
        c_dim=2 * (256 * 5),
        hidden_size=256,
        hidden_layers=5,
        output_ch=4,
        outermost_linear=True,
        first_omega_0=30.0,
        hidden_omega_0=30.0,
    ):
        super().__init__()
        if not outermost_linear:
            assert c_dim == 2 * (
                hidden_size * hidden_layers + output_ch
            ), "Incorrect c_dim in Siren!!!"
        else:
            assert c_dim == 2 * (
                hidden_size * hidden_layers
            ), "Incorrect c_dim in Siren!!!"

        self.hidden_size = hidden_size
        self.output_ch = output_ch

        self.net = nn.ModuleList()
        self.net.append(
            SineFiLMLayer(dim, hidden_size, is_first=True, omega_0=first_omega_0)
        )

        for i in range(hidden_layers):
            self.net.append(
                SineFiLMLayer(
                    hidden_size, hidden_size, is_first=False, omega_0=hidden_omega_0
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_size, output_ch)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_size) / hidden_omega_0,
                    np.sqrt(6 / hidden_size) / hidden_omega_0,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                SineFiLMLayer(
                    hidden_size, output_ch, is_first=False, omega_0=hidden_omega_0
                )
            )

    def forward(self, p, c, **kwargs):
        # shape of c is B * 2*(hidden_size*hidden_layers + output_ch)
        output = p  # coordinates
        split = int(c.shape[1] / 2)
        frequencies = c[:, :split] + 1.0
        shifts = c[:, split:]
        for i, layer in enumerate(self.net):
            # initial layer just encodes positions
            if i == 0 or (not isinstance(layer, SineFiLMLayer)):
                output = layer(output)
            else:
                f_i = frequencies[
                    :, (i - 1) * self.hidden_size : (i) * self.hidden_size
                ].unsqueeze(1)
                s_i = shifts[
                    :, (i - 1) * self.hidden_size : (i) * self.hidden_size
                ].unsqueeze(1)
                output = layer(output, f_i, s_i)
        output = output.squeeze(-1)
        return output


class ResidualLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.linear1 = nn.Linear(in_features, out_features, bias=bias)
        self.linear2 = nn.Linear(out_features, out_features, bias=bias)
        self.relu1 = nn.LeakyReLU(0.01, inplace=False)
        self.relu2 = nn.LeakyReLU(0.01, inplace=False)

    def forward(self, x_init):
        x = self.relu1(self.linear1(x_init))
        x = x_init + self.linear2(x)
        x = self.relu2(x)
        return x


class Resnet34ResFC(nn.Module):
    r"""ResNet-18 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    """

    def __init__(self, c_dim, normalize=True, use_linear=True, linear_dim=512):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet34(pretrained=True)
        self.features.fc = nn.Sequential()
        self.fc = nn.Sequential(
            nn.Linear(linear_dim, c_dim),
            nn.modules.normalization.LayerNorm(c_dim, elementwise_affine=False),
            nn.LeakyReLU(0.01, inplace=False),
            ResidualLayer(c_dim, c_dim),
        )

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


class VNet(nn.Module):
    """Volumetric Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
        p0_z (dist): prior distribution for latent code z
        device (device): torch device
    """

    def __init__(
        self,
        decoder=SirenFiLM(hidden_size=256, hidden_layers=5),
        encoder=Resnet34ResFC(c_dim=2560),
        device="cuda",
    ):
        super(VNet, self).__init__()

        self.decoder = decoder
        self.encoder = encoder
        self._device = device

    def forward(self):
        pass

    def to(self, device):
        """Puts the model to the device.

        Args:
            device (device): pytorch device
        """
        model = super().to(device)
        model._device = device
        return model
