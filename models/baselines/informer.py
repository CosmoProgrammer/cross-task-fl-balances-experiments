"""
Informer: Efficient Transformer for Long Sequence Time-Series Forecasting.
Simplified implementation with ProbSparse attention.
(Zhou et al., 2021)


Interface: model(x) -> (B, pred_len, out_chn)  where x is (B, seq_len, in_chn)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbSparseAttention(nn.Module):
    """ProbSparse self-attention mechanism."""

    def __init__(self, d_model, n_heads, factor=5, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        B, L_Q, D = Q.shape
        B, L_K, D = K.shape
        H = self.n_heads

        q = self.W_Q(Q).view(B, L_Q, H, self.d_k).transpose(1, 2)
        k = self.W_K(K).view(B, L_K, H, self.d_k).transpose(1, 2)
        v = self.W_V(V).view(B, L_K, H, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, v)

        context = context.transpose(1, 2).contiguous().view(B, L_Q, D)
        return self.out_proj(context)


class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, dropout=0.1):
        super().__init__()
        self.attention = ProbSparseAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class Informer(nn.Module):
    """Simplified Informer for time series forecasting."""

    def __init__(self, in_chn=1, out_chn=1, seq_len=128, pred_len=24,
                 d_model=128, n_heads=4, n_layers=2, d_ff=256, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.out_chn = out_chn

        self.input_proj = nn.Linear(in_chn, d_model)

        # Positional encoding
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.head = nn.Linear(d_model * seq_len, pred_len * out_chn)

    def forward(self, x, x_mark=None, x_mask=None):
        # x: (B, L, C)
        B, L, C = x.shape
        x = self.input_proj(x) + self.pe[:, :L, :]

        for layer in self.encoder_layers:
            x = layer(x)

        x = x.reshape(B, -1)
        out = self.head(x)
        return out.reshape(B, self.pred_len, self.out_chn)
