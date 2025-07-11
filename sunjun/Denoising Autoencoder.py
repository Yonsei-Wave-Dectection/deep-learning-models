import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class DenoisingAutoencoder1D(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[64, 32, 16], latent_dim=8):
        super(DenoisingAutoencoder1D, self).__init__()

        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Conv1d(in_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            in_dim = hidden_dim

        encoder_layers.append(nn.Conv1d(in_dim, latent_dim, kernel_size=3, stride=2, padding=1))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.ConvTranspose1d(in_dim, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            in_dim = hidden_dim

        decoder_layers.append(nn.ConvTranspose1d(in_dim, input_dim, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
