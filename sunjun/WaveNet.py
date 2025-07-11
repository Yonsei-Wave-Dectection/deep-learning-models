import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class WaveNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, residual_channels=64,
                 skip_channels=64, layers=10, stacks=3, kernel_size=2):
        super(WaveNet, self).__init__()

        self.layers = layers
        self.stacks = stacks

        # Input projection
        self.start_conv = nn.Conv1d(in_channels, residual_channels, kernel_size=1)

        # Dilated convolution layers
        self.dilated_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        for stack in range(stacks):
            for layer in range(layers):
                dilation = 2 ** layer
                self.dilated_convs.append(
                    nn.Conv1d(residual_channels, 2 * residual_channels,
                             kernel_size=kernel_size, dilation=dilation, padding=dilation)
                )
                self.residual_convs.append(
                    nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
                )
                self.skip_convs.append(
                    nn.Conv1d(residual_channels, skip_channels, kernel_size=1)
                )

        # Output layers
        self.end_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.end_conv2 = nn.Conv1d(skip_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.start_conv(x)
        skip_connections = []

        for i in range(self.stacks * self.layers):
            # Dilated convolution
            conv_out = self.dilated_convs[i](x)

            # Gated activation
            filter_out, gate_out = conv_out.chunk(2, dim=1)
            gated = torch.tanh(filter_out) * torch.sigmoid(gate_out)

            # Residual and skip connections
            residual = self.residual_convs[i](gated)
            skip = self.skip_convs[i](gated)
            skip_connections.append(skip)

            x = x + residual

        # Sum skip connections
        skip_sum = sum(skip_connections)

        # Output layers
        out = F.relu(self.end_conv1(skip_sum))
        out = self.end_conv2(out)

        return out
