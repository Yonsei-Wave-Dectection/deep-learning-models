import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class FFTformer(nn.Module):
    def __init__(self, seq_len=1024, d_model=64, nhead=8, num_layers=4, dropout=0.1):
        super(FFTformer, self).__init__()

        self.seq_len = seq_len
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout, seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # FFT processing
        self.fft_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)
        batch_size, channels, seq_len = x.shape

        # Reshape for transformer
        x = x.transpose(1, 2)  # (batch_size, sequence_length, channels)

        # Input projection
        x = self.input_proj(x)  # (batch_size, sequence_length, d_model)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply transformer
        x_transformed = self.transformer(x)

        # FFT processing in frequency domain
        x_fft = torch.fft.fft(x_transformed, dim=1)
        x_fft_real = torch.real(x_fft)
        x_fft_processed = self.fft_proj(x_fft_real)

        # Inverse FFT
        x_ifft = torch.fft.ifft(torch.complex(x_fft_processed, torch.zeros_like(x_fft_processed)), dim=1)
        x_reconstructed = torch.real(x_ifft)

        # Combine with original transformed features
        x_combined = x_transformed + x_reconstructed

        # Output projection
        output = self.output_proj(x_combined)

        # Transpose back to original format
        output = output.transpose(1, 2)  # (batch_size, channels, sequence_length)

        return output
