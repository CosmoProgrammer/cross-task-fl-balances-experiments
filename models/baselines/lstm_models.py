"""
LSTM models for both forecasting and anomaly detection (autoencoder).

Interface contract:
  Forecasting: model(x) -> (B, pred_len, out_chn)  where x is (B, seq_len, in_chn)
  Anomaly:     model(x) -> (B, seq_len, 1)          where x is (B, seq_len, 1)
"""
import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """LSTM for time series forecasting."""

    def __init__(self, in_chn=1, out_chn=1, seq_len=128, pred_len=24,
                 hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.pred_len = pred_len
        self.out_chn = out_chn

        self.lstm = nn.LSTM(
            input_size=in_chn, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, out_chn * pred_len)

    def forward(self, x, x_mark=None, x_mask=None):
        # x: (B, seq_len, in_chn)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = self.fc(out)
        return out.view(-1, self.pred_len, self.out_chn)


class LSTMAutoEncoder(nn.Module):
    """LSTM autoencoder for anomaly detection via reconstruction error.

    Accepts the same masked input as MambaMixer anomaly model.
    Does NOT use x_mask (standard denoising AE approach).
    """

    def __init__(self, seq_len=128, input_dim=1, hidden_dim=64, num_layers=2,
                 dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.expand = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, x_mask=None):
        # x: (B, seq_len, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        _, (h, c) = self.encoder(x)
        encoded = self.bottleneck(h[-1])
        decoder_input = self.expand(encoded)
        decoder_input = decoder_input.unsqueeze(1).repeat(1, self.seq_len, 1)
        decoder_out, _ = self.decoder(decoder_input, (h, c))
        reconstruction = self.output_layer(decoder_out)
        return reconstruction
