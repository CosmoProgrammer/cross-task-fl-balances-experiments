"""
ANN (feedforward) autoencoder for anomaly detection.

Interface: model(x) -> (B, seq_len, 1)  where x is (B, seq_len, 1)
"""
import torch.nn as nn


class ANNAutoEncoder(nn.Module):
    """Feedforward autoencoder for anomaly detection."""

    def __init__(self, seq_len=128, input_dim=1, hidden_dims=(64, 32, 16),
                 dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        flat_dim = seq_len * input_dim

        # Encoder
        enc_layers = []
        in_dim = flat_dim
        for h in hidden_dims:
            enc_layers.extend([
                nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        for h in reversed(hidden_dims[:-1]):
            dec_layers.extend([
                nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, flat_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x, x_mask=None):
        # x: (B, seq_len, 1)
        if x.dim() == 3:
            B, L, C = x.shape
            x_flat = x.reshape(B, -1)
        else:
            B, L = x.shape
            C = 1
            x_flat = x

        encoded = self.encoder(x_flat)
        decoded = self.decoder(encoded)
        return decoded.reshape(B, L, max(C, 1))
