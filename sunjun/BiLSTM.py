import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class BiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(BiLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, bidirectional=True, dropout=dropout)

        # Output layer (hidden_size * 2 because of bidirectional)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)
        # LSTM expects: (batch_size, sequence_length, features)
        x = x.transpose(1, 2)  # (batch_size, sequence_length, channels)

        # Initialize hidden states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)

        # Apply fully connected layer to each time step
        out = self.fc(out)

        # Transpose back to original format
        out = out.transpose(1, 2)  # (batch_size, channels, sequence_length)

        return out
