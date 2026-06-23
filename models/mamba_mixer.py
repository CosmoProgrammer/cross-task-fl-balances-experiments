"""
MambaMixer: State-Space Decomposition Network with MLP-Mixer.

Uses BiMamba (bidirectional selective SSM) with multi-scale patch
decomposition for energy time series analysis.

Key components:
  1. BiMambaBlock - Bidirectional selective state-space model
  2. CrossScaleGate - FiLM conditioning between scales
  3. AdaptiveScaleRouter - Input-dependent scale weighting
  4. Multi-scale patch encoder/decoder with SSM temporal mixing
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from einops import rearrange
from einops.layers.torch import Rearrange


# ── Optional fused CUDA kernels (mamba-ssm / causal-conv1d) ──
# Not installed on the dev laptop (no Windows build); the pure-PyTorch path
# below runs as a numerically-equivalent fallback when they are absent.
# Install on the Linux GPU server to activate -- see profile_speed.py and the
# server instructions. SelectiveSSM.forward / BiMambaBlock._scan dispatch to
# the kernel only when USE_KERNELS and the import succeeded and x.is_cuda.
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    _HAS_SELECTIVE_SCAN = True
except ImportError:
    selective_scan_fn = None
    _HAS_SELECTIVE_SCAN = False

try:
    from causal_conv1d import causal_conv1d_fn
    _HAS_CAUSAL_CONV = True
except ImportError:
    causal_conv1d_fn = None
    _HAS_CAUSAL_CONV = False

# Master switch; main.py sets this from config.use_mamba_kernel. When False the
# eager path runs even if the kernels import (to reproduce pre-kernel results or
# for the parity check in test_kernel_parity.py).
USE_KERNELS = True


def _maybe_checkpoint(fn, h):
    """Gradient-checkpoint the SSM block ONLY on the eager path.

    Checkpointing exists to avoid retaining the eager Hillis-Steele scan's huge
    autograd graph (without it the eager path is ~5x slower). The fused kernel
    never materializes that graph, so checkpoint recompute is pure overhead there
    -- disabling it is ~1.5x faster (measured on the 4090). Gate matches the
    kernel dispatch in SelectiveSSM.forward so the two stay consistent.
    """
    if USE_KERNELS and _HAS_SELECTIVE_SCAN and h.is_cuda:
        return fn(h)
    return grad_checkpoint(fn, h, use_reentrant=False)


def get_activation(name):
    return {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh()}[name]


# ── MLP Block (MSD-Mixer style) ──

class MLPBlock(nn.Module):
    def __init__(self, dim, in_features, hid_features, out_features,
                 activ="gelu", drop=0.0, jump_conn="trunc"):
        super().__init__()
        self.dim = dim
        self.out_features = out_features
        self.net = nn.Sequential(
            nn.Linear(in_features, hid_features),
            get_activation(activ),
            nn.Linear(hid_features, out_features),
            nn.Dropout(drop),
        )
        if jump_conn == "trunc":
            self.jump_net = nn.Identity()
        elif jump_conn == "proj":
            self.jump_net = nn.Linear(in_features, out_features)
        else:
            raise ValueError(f"jump_conn: {jump_conn}")

    def forward(self, x):
        x = torch.transpose(x, self.dim, -1)
        x = self.jump_net(x)[..., :self.out_features] + self.net(x)
        x = torch.transpose(x, self.dim, -1)
        return x


# ── Selective State-Space Model (Mamba S6 core) ──

class SelectiveSSM(nn.Module):
    def __init__(self, d_inner, state_size=16, dt_rank=None):
        super().__init__()
        self.d_inner = d_inner
        self.state_size = state_size
        if dt_rank is None:
            dt_rank = max(d_inner // 16, 1)
        self.dt_rank = dt_rank

        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, state_size + 1, dtype=torch.float32)
                      .unsqueeze(0).expand(d_inner, -1).clone())
        )
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * state_size, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        self.D = nn.Parameter(torch.ones(d_inner))

        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(d_inner) * (math.log(0.1) - math.log(0.001))
                + math.log(0.001))
            inv_sp = dt_init + torch.log(-torch.expm1(-dt_init))
            self.dt_proj.bias.copy_(inv_sp)

    @staticmethod
    def _parallel_scan_simple(A_bar, Bu):
        """Parallel scan using iterative doubling (Hillis-Steele style).

        Simpler and more numerically stable than Blelloch for GPU execution.
        O(L log L) work but O(log L) sequential steps instead of O(L).

        Args:
            A_bar: (B, L, D, N) - discretized state transition
            Bu:    (B, L, D, N) - input contribution

        Returns:
            h: (B, L, D, N) - hidden states at all timesteps
        """
        # h[t] = A_bar[t] * h[t-1] + Bu[t]
        # Represent as (a, b) tuples where h = a * h_init + b
        # Initially: a[t] = A_bar[t], b[t] = Bu[t]
        # Combine (a2, b2) . (a1, b1) = (a2*a1, a2*b1 + b2)
        # After log2(L) steps of doubling, b contains the scan result (since h_init=0)

        a = A_bar.clone()
        b = Bu.clone()
        L = a.shape[1]

        stride = 1
        while stride < L:
            a_shifted = F.pad(a[:, :-stride], (0, 0, 0, 0, stride, 0), value=1.0)
            b_shifted = F.pad(b[:, :-stride], (0, 0, 0, 0, stride, 0), value=0.0)
            new_b = a * b_shifted + b
            new_a = a * a_shifted
            a = new_a
            b = new_b
            stride *= 2

        return b  # h[t] = a[t]*0 + b[t] = b[t] since h_init = 0

    def forward(self, u):
        # Dispatch to the fused kernel when available; otherwise eager scan.
        # Numerically equivalent up to fp tolerance (verified by
        # test_kernel_parity.py). The silu z-gate stays in BiMambaBlock, so it
        # is not folded in here.
        if USE_KERNELS and _HAS_SELECTIVE_SCAN and u.is_cuda:
            return self._kernel_forward(u)
        return self._eager_forward(u)

    def _eager_forward(self, u):
        B, L, D = u.shape
        N = self.state_size

        x_dbl = self.x_proj(u)
        dt_raw, B_sel, C_sel = x_dbl.split([self.dt_rank, N, N], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))          # (B, L, D)
        A = -torch.exp(self.A_log)                      # (D, N)

        # Discretize: A_bar[t] = exp(dt[t] * A), Bu[t] = (dt[t] * u[t]) * B_sel[t]
        A_bar = torch.exp(dt.unsqueeze(-1) * A)         # (B, L, D, N)
        dB_u = (dt * u).unsqueeze(-1) * B_sel.unsqueeze(2)  # (B, L, D, N)

        # Parallel scan: replaces the sequential for-loop over L timesteps
        # O(log L) sequential steps instead of O(L)
        h = self._parallel_scan_simple(A_bar, dB_u)     # (B, L, D, N)

        # Output: y[t] = h[t] . C_sel[t]
        y = torch.einsum("bldn,bln->bld", h, C_sel)

        return y + u * self.D

    def _kernel_forward(self, u):
        """Fused selective scan (mamba-ssm). Same recurrence as the eager path:
        A_bar=exp(dt*A), dB_u=dt*B*u, h[t]=A_bar*h[t-1]+dB_u, y=C.h + D*u.

        dt_proj's bias is passed as ``delta_bias`` with ``delta_softplus=True``
        so the kernel reproduces eager's ``softplus(dt_proj(dt_raw))`` exactly.
        A / D / delta_bias are kept in fp32 as the kernel requires.
        """
        N = self.state_size
        x_dbl = self.x_proj(u)
        dt_raw, B_sel, C_sel = x_dbl.split([self.dt_rank, N, N], dim=-1)
        delta = F.linear(dt_raw, self.dt_proj.weight)   # (B, L, D), bias deferred
        A = -torch.exp(self.A_log.float())              # (D, N)

        y = selective_scan_fn(
            u.transpose(1, 2).contiguous(),             # (B, D, L)
            delta.transpose(1, 2).contiguous(),         # (B, D, L)
            A,                                          # (D, N)
            B_sel.transpose(1, 2).contiguous(),         # (B, N, L)
            C_sel.transpose(1, 2).contiguous(),         # (B, N, L)
            D=self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )                                               # (B, D, L)
        return y.transpose(1, 2)                         # (B, L, D)


# ── Bidirectional Mamba Block ──

class BiMambaBlock(nn.Module):
    def __init__(self, d_model, state_size=16, expand=2, dt_rank=None,
                 conv_kernel=4, drop=0.0):
        super().__init__()
        d_inner = d_model * expand
        self.d_inner = d_inner
        self.norm = nn.LayerNorm(d_model)

        self.fwd_in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.fwd_conv = nn.Conv1d(d_inner, d_inner, conv_kernel,
                                  padding=conv_kernel - 1, groups=d_inner)
        self.fwd_ssm = SelectiveSSM(d_inner, state_size, dt_rank)

        self.bwd_in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.bwd_conv = nn.Conv1d(d_inner, d_inner, conv_kernel,
                                  padding=conv_kernel - 1, groups=d_inner)
        self.bwd_ssm = SelectiveSSM(d_inner, state_size, dt_rank)

        self.out_proj = nn.Linear(2 * d_inner, d_model, bias=False)
        self.drop = nn.Dropout(drop)

    def _scan(self, u, in_proj, conv, ssm, reverse=False):
        if reverse:
            u = u.flip(1)
        xz = in_proj(u)
        x, z = xz.chunk(2, dim=-1)
        L = x.shape[1]
        xt = x.transpose(1, 2)                          # (B, d_inner, L)
        if USE_KERNELS and _HAS_CAUSAL_CONV and x.is_cuda:
            # Fused causal depthwise conv + silu (kernel left-pads internally).
            xt = causal_conv1d_fn(xt, conv.weight.squeeze(1), conv.bias,
                                  activation="silu")
            x = xt.transpose(1, 2)
        else:
            x = conv(xt)[:, :, :L].transpose(1, 2)
            x = F.silu(x)
        x = ssm(x)
        x = x * F.silu(z)
        if reverse:
            x = x.flip(1)
        return x

    def forward(self, u):
        residual = u
        u = self.norm(u)
        x_fwd = self._scan(u, self.fwd_in_proj, self.fwd_conv, self.fwd_ssm, False)
        x_bwd = self._scan(u, self.bwd_in_proj, self.bwd_conv, self.bwd_ssm, True)
        x = self.out_proj(torch.cat([x_fwd, x_bwd], dim=-1))
        return self.drop(x) + residual


# ── Cross-Scale Gate (FiLM conditioning) ──

class CrossScaleGate(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.proj = nn.Linear(n_channels, n_channels * 2)

    def forward(self, emb, prev_summary):
        if prev_summary is None:
            return emb
        gb = self.proj(prev_summary)
        gate, bias = gb.chunk(2, dim=-1)
        return emb * torch.sigmoid(gate).unsqueeze(-1) + bias.unsqueeze(-1)


# ── Adaptive Scale Router ──

class AdaptiveScaleRouter(nn.Module):
    def __init__(self, in_chn, n_scales):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(in_chn, in_chn), nn.GELU(), nn.Linear(in_chn, n_scales))

    def forward(self, x):
        return F.softmax(self.router(x.mean(dim=-1)), dim=-1)


# ── SSM Patch Encoder ──

class SSMPatchEncoder(nn.Module):
    def __init__(self, in_len, hid_len, in_chn, hid_chn, out_chn,
                 patch_size, hid_pch, d_ssm=32, state_size=16,
                 expand=2, conv_kernel=4, norm=None, activ="gelu", drop=0.0):
        super().__init__()
        n_patches = in_len // patch_size
        self.patch_size = patch_size
        self.out_chn = out_chn

        norm_class = {"bn": nn.BatchNorm2d, "in": nn.InstanceNorm2d}.get(norm, nn.Identity)
        self.rearrange_in = Rearrange("b c (l1 l2) -> b c l1 l2", l2=patch_size)
        self.norm1 = norm_class(in_chn)
        self.channel_mlp = MLPBlock(1, in_chn, hid_chn, out_chn, activ, drop)

        self.norm2 = norm_class(out_chn)
        self.pre_ssm = nn.Linear(patch_size, d_ssm)
        self.bi_mamba = BiMambaBlock(d_ssm, state_size, expand,
                                     conv_kernel=conv_kernel, drop=drop)
        self.post_ssm = nn.Linear(d_ssm, patch_size)

        self.norm3 = norm_class(out_chn)
        self.intra_patch_mlp = MLPBlock(3, patch_size, hid_pch, patch_size, activ, drop)
        self.collapse = nn.Linear(patch_size, 1)

    def _ssm_forward(self, h):
        h = self.pre_ssm(h)
        h = self.bi_mamba(h)
        h = self.post_ssm(h)
        return h

    def forward(self, x):
        x = self.rearrange_in(x)
        x = self.channel_mlp(self.norm1(x))

        x = self.norm2(x)
        B, C, L1, L2 = x.shape
        h = x.reshape(B * C, L1, L2)
        h = _maybe_checkpoint(self._ssm_forward, h)
        x = x + h.reshape(B, C, L1, L2)

        x = self.intra_patch_mlp(self.norm3(x))
        return self.collapse(x).squeeze(-1)


# ── SSM Patch Decoder ──

class SSMPatchDecoder(nn.Module):
    def __init__(self, in_len, hid_len, in_chn, hid_chn, out_chn,
                 patch_size, hid_pch, d_ssm=32, state_size=16,
                 expand=2, conv_kernel=4, norm=None, activ="gelu", drop=0.0):
        super().__init__()
        self.patch_size = patch_size
        norm_class = {"bn": nn.BatchNorm2d, "in": nn.InstanceNorm2d}.get(norm, nn.Identity)

        self.expand_layer = nn.Linear(1, patch_size)
        self.norm1 = norm_class(in_chn)
        self.intra_patch_mlp = MLPBlock(3, patch_size, hid_pch, patch_size, activ, drop)

        self.norm2 = norm_class(in_chn)
        self.pre_ssm = nn.Linear(patch_size, d_ssm)
        self.bi_mamba = BiMambaBlock(d_ssm, state_size, expand,
                                     conv_kernel=conv_kernel, drop=drop)
        self.post_ssm = nn.Linear(d_ssm, patch_size)

        self.norm3 = norm_class(in_chn)
        self.channel_mlp = MLPBlock(1, in_chn, hid_chn, out_chn, activ, drop)
        self.rearrange_out = Rearrange("b c l1 l2 -> b c (l1 l2)")

    def _ssm_forward(self, h):
        h = self.pre_ssm(h)
        h = self.bi_mamba(h)
        h = self.post_ssm(h)
        return h

    def forward(self, x):
        x = self.expand_layer(x.unsqueeze(-1))
        x = self.intra_patch_mlp(self.norm1(x))

        x = self.norm2(x)
        B, C, L1, L2 = x.shape
        h = x.reshape(B * C, L1, L2)
        h = _maybe_checkpoint(self._ssm_forward, h)
        x = x + h.reshape(B, C, L1, L2)

        x = self.channel_mlp(self.norm3(x))
        return self.rearrange_out(x)


# ── Prediction Head ──

class PredictionHead(nn.Module):
    def __init__(self, in_len, out_len, hid_len, in_chn, out_chn, hid_chn,
                 activ, drop=0.0):
        super().__init__()
        c_jump = "proj" if in_chn != out_chn else "trunc"
        self.net = nn.Sequential(
            MLPBlock(1, in_chn, hid_chn, out_chn, activ=activ, drop=drop,
                     jump_conn=c_jump),
            MLPBlock(2, in_len, hid_len, out_len, activ=activ, drop=drop,
                     jump_conn="proj"),
        )

    def forward(self, x):
        return self.net(x)


# ── Residual Loss ──

def autocorrelation(x, dim=0, eps=0):
    N = x.size(dim)
    M = N
    while True:
        remaining = M
        for n in (2, 3, 5):
            while remaining % n == 0:
                remaining //= n
        if remaining == 1:
            break
        M += 1
    M2 = 2 * M
    x = x.transpose(dim, -1)
    centered = x - x.mean(dim=-1, keepdim=True)
    freqvec = torch.view_as_real(torch.fft.rfft(centered, n=M2))
    gram = freqvec.pow(2).sum(-1)
    acorr = torch.fft.irfft(gram, n=M2)
    acorr = acorr[..., :N]
    acorr = acorr / torch.arange(N, 0, -1, dtype=x.dtype, device=x.device)
    acorr = acorr / (acorr[..., :1] + eps)
    return acorr.transpose(dim, -1)


def residual_loss_fn(res, lambda_mse, lambda_acf, acf_cutoff=2, eps=0):
    import numpy as np
    loss = torch.tensor(0.0, device=res.device)
    if lambda_mse != 0:
        loss = loss + lambda_mse * torch.pow(res, 2).mean()
    if lambda_acf != 0:
        res_acf = F.relu(
            torch.abs(autocorrelation(res, -1, eps)[:, :, 1:])
            - acf_cutoff / np.sqrt(res.shape[-1]))
        loss = loss + lambda_acf * torch.pow(res_acf, 2).mean()
    return loss


# ═══════════════════════════════════════════════════════════════
#  MambaMixer
# ═══════════════════════════════════════════════════════════════

class MambaMixer(nn.Module):
    """
    MambaMixer.

    Modes:
      - Forecasting: out_len=24, out_chn=1 → returns y_pred
      - Reconstruction (anomaly): out_len=seq_len, out_chn=1 → returns y_pred
        (pred_heads predict the input itself)
      - Pure reconstruction: out_len=0, out_chn=0 → returns recon = x_orig - residual

    forward(x, x_mark, x_mask) -> tensor
      x: (B, L, in_chn), x_mark: (B, L, ex_chn) or None, x_mask: (B, L, in_chn) or None
    """

    # Parameter prefixes for backbone vs task heads
    BACKBONE_PREFIXES = ("patch_encoders.", "patch_decoders.", "cross_scale_gates.")
    HEAD_PREFIXES = ("pred_heads.", "scale_router.")

    def __init__(self, in_len, out_len, in_chn, ex_chn, out_chn,
                 patch_sizes=(24, 12, 6, 2, 1), hid_len=128, hid_chn=256,
                 hid_pch=64, hid_pred=128, d_ssm=64, state_size=32,
                 expand=2, conv_kernel=4, norm=None, last_norm=True,
                 activ="gelu", drop=0.15):
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.last_norm = last_norm
        self.patch_sizes = list(patch_sizes)

        self.patch_encoders = nn.ModuleList()
        self.patch_decoders = nn.ModuleList()
        self.pred_heads = nn.ModuleList()
        self.cross_scale_gates = nn.ModuleList()
        self.paddings = []

        all_chn = in_chn + ex_chn
        ssm_kw = dict(d_ssm=d_ssm, state_size=state_size,
                       expand=expand, conv_kernel=conv_kernel)

        for ps in self.patch_sizes:
            padding = (ps - in_len % ps) % ps
            self.paddings.append(padding)
            padded = in_len + padding

            self.patch_encoders.append(
                SSMPatchEncoder(padded, hid_len, all_chn, hid_chn, in_chn,
                                ps, hid_pch, norm=norm, activ=activ, drop=drop,
                                **ssm_kw))
            self.patch_decoders.append(
                SSMPatchDecoder(padded, hid_len, in_chn, hid_chn, in_chn,
                                ps, hid_pch, norm=norm, activ=activ, drop=drop,
                                **ssm_kw))
            if out_len and out_chn:
                self.pred_heads.append(
                    PredictionHead(padded // ps, out_len, hid_pred,
                                   in_chn, out_chn, hid_chn, activ, drop))
            else:
                self.pred_heads.append(nn.Identity())
            self.cross_scale_gates.append(CrossScaleGate(in_chn))

        if out_len and out_chn:
            self.scale_router = AdaptiveScaleRouter(in_chn, len(self.patch_sizes))

    @staticmethod
    def is_backbone_param(name):
        """Check if a parameter name belongs to the shared backbone."""
        return any(name.startswith(p) for p in MambaMixer.BACKBONE_PREFIXES)

    @staticmethod
    def is_head_param(name):
        """Check if a parameter name belongs to a task-specific head."""
        return any(name.startswith(p) for p in MambaMixer.HEAD_PREFIXES)

    def forward(self, x, x_mark=None, x_mask=None):
        x = rearrange(x, "b l c -> b c l")
        if x_mark is not None:
            x_mark = rearrange(x_mark, "b l c -> b c l")
        if x_mask is not None:
            x_mask = rearrange(x_mask, "b l c -> b c l")

        if self.last_norm:
            x_last = x[:, :, [-1]].detach()
            x = x - x_last
            if x_mark is not None:
                x_mark_last = x_mark[:, :, [-1]].detach()
                x_mark = x_mark - x_mark_last

        x_orig = x
        y_pred = []
        prev_summary = None

        for i in range(len(self.patch_sizes)):
            x_in = torch.cat((x, x_mark), 1) if x_mark is not None else x
            x_in = F.pad(x_in, (self.paddings[i], 0), "constant", 0)

            emb = self.patch_encoders[i](x_in)
            emb = self.cross_scale_gates[i](emb, prev_summary)
            prev_summary = emb.mean(dim=-1)

            comp = self.patch_decoders[i](emb)[:, :, self.paddings[i]:]
            pred = self.pred_heads[i](emb)

            if x_mask is not None:
                comp = comp * x_mask
            x = x - comp

            if self.out_len and self.out_chn:
                y_pred.append(pred)

        # Store residual for auxiliary loss
        self.last_residual = x

        if self.out_len and self.out_chn:
            stacked = torch.stack(y_pred, dim=0)
            scale_weights = self.scale_router(x_orig)
            y_pred = torch.einsum("sbcl,bs->bcl", stacked, scale_weights)

            if self.last_norm and self.out_chn == self.in_chn:
                y_pred = y_pred + x_last
            y_pred = rearrange(y_pred, "b c l -> b l c")
            return y_pred

        # Reconstruction mode: return reconstructed input
        recon = rearrange(x_orig - x, "b c l -> b l c")
        if self.last_norm and self.out_chn == self.in_chn:
            recon = recon + rearrange(x_last, "b c l -> b l c")
        return recon
