"""Parity check: fused mamba-ssm/causal-conv1d kernels vs the pure-PyTorch
eager path. Run this ON THE SERVER (where the kernels are installed) BEFORE
trusting any results produced with use_mamba_kernel=True -- the kernel uses a
different algorithm, so it is only numerically equivalent up to fp tolerance.

    conda run -n crosstask python test_kernel_parity.py

SKIPs gracefully if the kernels are not importable (e.g. the dev laptop).
Compares forward outputs and all parameter gradients; prints max abs/rel diff.
"""
import torch

import models.mamba_mixer as mm
from models.mamba_mixer import MambaMixer
from configs.config import ExperimentConfig


def build(config, task):
    out_len = config.pred_len if task == "forecasting" else config.seq_len
    return MambaMixer(
        in_len=config.seq_len, out_len=out_len,
        in_chn=config.in_chn, ex_chn=config.ex_chn, out_chn=config.out_chn,
        patch_sizes=config.patch_sizes, hid_len=config.hid_len,
        hid_chn=config.hid_chn, hid_pch=config.hid_pch,
        hid_pred=config.hid_pred, d_ssm=config.d_ssm,
        state_size=config.state_size, expand=config.expand,
        conv_kernel=config.conv_kernel, last_norm=config.last_norm,
        drop=config.drop,
    )


def run_once(model, x, mask, task, use_kernels):
    mm.USE_KERNELS = use_kernels
    model.zero_grad(set_to_none=True)
    out = model(x, x_mask=mask) if task == "anomaly" else model(x)
    out.sum().backward()
    grads = {n: p.grad.detach().clone()
             for n, p in model.named_parameters() if p.grad is not None}
    return out.detach().clone(), grads


def max_diff(a, b):
    abs_d = (a - b).abs().max().item()
    rel_d = ((a - b).abs() / (b.abs() + 1e-6)).max().item()
    return abs_d, rel_d


def check(task, config, device):
    print(f"\n=== {task} ===")
    torch.manual_seed(0)
    model = build(config, task).to(device).eval()  # eval() -> no dropout noise
    B, L, C = 8, config.seq_len, config.in_chn
    x = torch.randn(B, L, C, device=device)
    mask = ((torch.rand(B, L, C, device=device) > config.mask_rate).float()
            if task == "anomaly" else None)

    out_e, grad_e = run_once(model, x, mask, task, use_kernels=False)
    out_k, grad_k = run_once(model, x, mask, task, use_kernels=True)

    a, r = max_diff(out_k, out_e)
    print(f"  output      max|abs|={a:.2e}  max|rel|={r:.2e}")
    worst = (0.0, None)
    for n in grad_e:
        if n not in grad_k:
            print(f"  !! grad missing in kernel path: {n}")
            continue
        ga, _ = max_diff(grad_k[n], grad_e[n])
        if ga > worst[0]:
            worst = (ga, n)
    print(f"  grads       max|abs|={worst[0]:.2e}  (worst: {worst[1]})")
    ok = a < 2e-3 and worst[0] < 5e-3
    print(f"  -> {'PASS' if ok else 'CHECK -- diffs larger than expected'}")
    return ok


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA.")
        return
    if not mm._HAS_SELECTIVE_SCAN:
        print("SKIP: mamba-ssm (selective_scan_fn) not installed -- nothing to "
              "compare. Install on the server and re-run.")
        return
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"selective_scan={mm._HAS_SELECTIVE_SCAN}  "
          f"causal_conv1d={mm._HAS_CAUSAL_CONV}")
    config = ExperimentConfig()
    ok = all([check("forecasting", config, device),
              check("anomaly", config, device)])
    print(f"\n{'ALL PASS' if ok else 'SOME CHECKS FLAGGED -- inspect above'}")


if __name__ == "__main__":
    main()
