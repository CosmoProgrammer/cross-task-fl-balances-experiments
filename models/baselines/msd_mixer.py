"""
MSD-Mixer: Multi-Scale Decomposition MLP-Mixer for Time Series Analysis.
(Zhong et al., VLDB 2024)
  Forecasting: model(x) -> (B, pred_len, out_chn)
  Anomaly:     model(x) -> (B, seq_len, 1)

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


def _get_activation(name):
    return {"gelu": nn.GELU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh()}[name]


class _MLPBlock(nn.Module):
    def __init__(self, dim, in_features, hid_features, out_features,
                 activ="gelu", drop=0.0, jump_conn="trunc"):
        super().__init__()
        self.dim = dim
        self.out_features = out_features
        self.net = nn.Sequential(
            nn.Linear(in_features, hid_features),
            _get_activation(activ),
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


class _PatchEncoder(nn.Module):
    """MLP-only patch encoder (no SSM)."""

    def __init__(self, in_len, hid_len, in_chn, hid_chn, out_chn,
                 patch_size, hid_pch, norm=None, activ="gelu", drop=0.0):
        super().__init__()
        n_patches = in_len // patch_size

        norm_class = {"bn": nn.BatchNorm2d, "in": nn.InstanceNorm2d
                      }.get(norm, nn.Identity)

        self.net = nn.Sequential(
            Rearrange("b c (l1 l2) -> b c l1 l2", l2=patch_size),
            norm_class(in_chn),
            _MLPBlock(1, in_chn, hid_chn, out_chn, activ, drop),
            norm_class(out_chn),
            _MLPBlock(2, n_patches, hid_len, n_patches, activ, drop),
            norm_class(out_chn),
            _MLPBlock(3, patch_size, hid_pch, patch_size, activ, drop),
            nn.Linear(patch_size, 1),
            Rearrange("b c l1 1 -> b c l1"),
        )

    def forward(self, x):
        return self.net(x)


class _PatchDecoder(nn.Module):
    """MLP-only patch decoder (no SSM)."""

    def __init__(self, in_len, hid_len, in_chn, hid_chn, out_chn,
                 patch_size, hid_pch, norm=None, activ="gelu", drop=0.0):
        super().__init__()
        n_patches = in_len // patch_size

        norm_class = {"bn": nn.BatchNorm2d, "in": nn.InstanceNorm2d
                      }.get(norm, nn.Identity)

        self.net = nn.Sequential(
            Rearrange("b c l1 -> b c l1 1"),
            nn.Linear(1, patch_size),
            norm_class(in_chn),
            _MLPBlock(3, patch_size, hid_pch, patch_size, activ, drop),
            norm_class(in_chn),
            _MLPBlock(2, n_patches, hid_len, n_patches, activ, drop),
            norm_class(in_chn),
            _MLPBlock(1, in_chn, hid_chn, out_chn, activ, drop),
            Rearrange("b c l1 l2 -> b c (l1 l2)"),
        )

    def forward(self, x):
        return self.net(x)


class _PredictionHead(nn.Module):
    def __init__(self, in_len, out_len, hid_len, in_chn, out_chn, hid_chn,
                 activ, drop=0.0):
        super().__init__()
        c_jump = "proj" if in_chn != out_chn else "trunc"
        self.net = nn.Sequential(
            _MLPBlock(1, in_chn, hid_chn, out_chn, activ=activ, drop=drop,
                      jump_conn=c_jump),
            _MLPBlock(2, in_len, hid_len, out_len, activ=activ, drop=drop,
                      jump_conn="proj"),
        )

    def forward(self, x):
        return self.net(x)


class MSDMixer(nn.Module):
    """MSD-Mixer: Multi-Scale Decomposition MLP-Mixer.


    Interface matches MambaMixer:
      Forecasting:     model(x) -> (B, pred_len, out_chn)
      Reconstruction:  model(x) -> (B, seq_len, out_chn)
    """

    def __init__(self, in_len, out_len, in_chn=1, ex_chn=0, out_chn=1,
                 patch_sizes=(24, 12, 6, 2, 1), hid_len=128, hid_chn=256,
                 hid_pch=64, hid_pred=128, norm=None, last_norm=True,
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
        self.paddings = []

        all_chn = in_chn + ex_chn

        for ps in self.patch_sizes:
            padding = (ps - in_len % ps) % ps
            self.paddings.append(padding)
            padded = in_len + padding

            self.patch_encoders.append(
                _PatchEncoder(padded, hid_len, all_chn, hid_chn, in_chn,
                              ps, hid_pch, norm, activ, drop))
            self.patch_decoders.append(
                _PatchDecoder(padded, hid_len, in_chn, hid_chn, in_chn,
                              ps, hid_pch, norm, activ, drop))
            if out_len and out_chn:
                self.pred_heads.append(
                    _PredictionHead(padded // ps, out_len, hid_pred,
                                    in_chn, out_chn, hid_chn, activ, drop))
            else:
                self.pred_heads.append(nn.Identity())

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

        for i in range(len(self.patch_sizes)):
            x_in = torch.cat((x, x_mark), 1) if x_mark is not None else x
            x_in = F.pad(x_in, (self.paddings[i], 0), "constant", 0)

            emb = self.patch_encoders[i](x_in)
            comp = self.patch_decoders[i](emb)[:, :, self.paddings[i]:]
            pred = self.pred_heads[i](emb)

            if x_mask is not None:
                comp = comp * x_mask
            x = x - comp

            if self.out_len and self.out_chn:
                y_pred.append(pred)

        # Store residual for potential auxiliary loss
        self.last_residual = x

        if self.out_len and self.out_chn:
            # Sum reduction (original MSD-Mixer approach, no adaptive router)
            y_pred = torch.stack(y_pred, dim=0).sum(dim=0)

            if self.last_norm and self.out_chn == self.in_chn:
                y_pred = y_pred + x_last
            y_pred = rearrange(y_pred, "b c l -> b l c")
            return y_pred

        # Reconstruction mode
        recon = rearrange(x_orig - x, "b c l -> b l c")
        if self.last_norm and self.out_chn == self.in_chn:
            recon = recon + rearrange(x_last, "b c l -> b l c")
        return recon
