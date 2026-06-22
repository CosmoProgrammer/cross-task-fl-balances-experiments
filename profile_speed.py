"""Scratch profiler: attribute per-step training cost and measure the
free-optimization speedup ratio (TF32 / bf16 autocast / no grad-checkpoint).

Runs a handful of forward+backward+step iterations on synthetic batches that
match the real shapes -- it does NOT run an FL round. Absolute ms are
device-specific; the *ratios* and the op-breakdown transfer across Ada GPUs.

    conda run -n crosstask python profile_speed.py
    conda run -n crosstask python profile_speed.py --batch 64 --profile

Safe to delete; nothing imports it.
"""
import argparse
import time
import contextlib

import torch
import torch.nn as nn

import models.mamba_mixer as mm
from models.mamba_mixer import MambaMixer, SelectiveSSM
from configs.config import ExperimentConfig


# ── Memory-light sequential scan (avoids the Hillis-Steele padded
#    intermediates: L small steps over (B,D,N) instead of log L huge
#    pad/clone ops over (B,L,D,N)). Same recurrence, no new deps. ──
_USE_ALTSCAN = False
_orig_scan = SelectiveSSM._parallel_scan_simple


def _seq_scan(A_bar, Bu):
    L = A_bar.shape[1]
    h = Bu[:, 0]
    out = [h]
    for t in range(1, L):
        h = A_bar[:, t] * h + Bu[:, t]
        out.append(h)
    return torch.stack(out, dim=1)


def _scan_dispatch(A_bar, Bu):
    return _seq_scan(A_bar, Bu) if _USE_ALTSCAN else _orig_scan(A_bar, Bu)


SelectiveSSM._parallel_scan_simple = staticmethod(_scan_dispatch)


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


def make_batch(config, task, device):
    B, L, C = config.batch_size, config.seq_len, config.in_chn
    x = torch.randn(B, L, C, device=device)
    if task == "forecasting":
        y = torch.randn(B, config.pred_len, config.out_chn, device=device)
        return x, y, None
    mask = (torch.rand(B, L, C, device=device) > config.mask_rate).float()
    return x * mask, x, mask  # (input, target, mask)


# Module-level switch the (possibly patched) grad_checkpoint reads.
_USE_CKPT = True
_orig_ckpt = mm.grad_checkpoint


def _maybe_ckpt(fn, *a, **k):
    if _USE_CKPT:
        return _orig_ckpt(fn, *a, **k)
    return fn(*a)


mm.grad_checkpoint = _maybe_ckpt


def run_step(model, batch, task, opt, use_amp):
    x, y, mask = batch
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if use_amp else contextlib.nullcontext())
    opt.zero_grad(set_to_none=True)
    with autocast:
        out = model(x, x_mask=mask) if task == "anomaly" else model(x)
        if task == "anomaly":
            per = nn.functional.mse_loss(out, y, reduction="none")
            mpos = (mask == 0)
            loss = per[mpos].mean() if mpos.any() else per.mean()
        else:
            loss = nn.functional.mse_loss(out, y)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss


def time_condition(config, task, device, *, tf32, amp, ckpt, compile=False,
                   altscan=False, kernel=False, warmup=10, iters=30):
    global _USE_CKPT, _USE_ALTSCAN
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    _USE_CKPT = ckpt
    _USE_ALTSCAN = altscan
    mm.USE_KERNELS = kernel  # dispatch to fused mamba-ssm kernel when installed

    torch.manual_seed(0)
    model = build(config, task).to(device).train()
    if compile:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    batch = make_batch(config, task, device)

    for _ in range(warmup):
        run_step(model, batch, task, opt, amp)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run_step(model, batch, task, opt, amp)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    del model, opt
    torch.cuda.empty_cache()
    return ms


def profile_ops(config, task, device):
    from torch.profiler import profile, ProfilerActivity
    global _USE_CKPT
    _USE_CKPT = True
    mm.USE_KERNELS = False  # breakdown is about the eager scan
    torch.backends.cuda.matmul.allow_tf32 = False
    model = build(config, task).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = make_batch(config, task, device)
    for _ in range(5):
        run_step(model, batch, task, opt, False)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            run_step(model, batch, task, opt, False)
        torch.cuda.synchronize()
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--task", default="both",
                    choices=["forecasting", "anomaly", "both"])
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    config = ExperimentConfig()
    if args.batch:
        config.batch_size = args.batch
    assert torch.cuda.is_available(), "need CUDA"
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)} | batch={config.batch_size}")
    print(f"torch {torch.__version__}\n")

    tasks = ["forecasting", "anomaly"] if args.task == "both" else [args.task]

    if args.sweep:
        # Batch scaling: launch-bound work should give ~linear throughput gains.
        for task in tasks:
            print(f"=== {task}: batch sweep (fp32, ckpt) ===")
            base_per = None
            for B in [32, 64, 128, 256]:
                config.batch_size = B
                try:
                    ms = time_condition(config, task, device,
                                        tf32=False, amp=False, ckpt=True,
                                        warmup=5, iters=15)
                except RuntimeError as e:
                    print(f"  batch {B:4d}: OOM/err ({str(e)[:40]})")
                    torch.cuda.empty_cache()
                    continue
                per = ms / B
                if base_per is None:
                    base_per = per
                print(f"  batch {B:4d}: {ms:7.1f} ms/step  "
                      f"{per:6.2f} ms/sample  {base_per/per:4.2f}x throughput")
            print()
        return

    kernel_ok = mm._HAS_SELECTIVE_SCAN
    conditions = [
        ("eager scan (current)",       dict(tf32=False, amp=False, ckpt=True,
                                            kernel=False)),
    ]
    if kernel_ok:
        conditions += [
            ("mamba kernel (ckpt)",     dict(tf32=True, amp=False, ckpt=True,
                                            kernel=True)),
            ("mamba kernel -ckpt",      dict(tf32=True, amp=False, ckpt=False,
                                            kernel=True)),
        ]
    else:
        print("  (mamba-ssm not installed -- kernel rows skipped; "
              "install on the server to compare)")
    if args.compile:
        conditions.append(
            ("torch.compile (eager scan)",
             dict(tf32=True, amp=False, ckpt=True, compile=True, kernel=False)))

    for task in tasks:
        print(f"=== {task} ===")
        base = None
        for name, kw in conditions:
            try:
                ms = time_condition(config, task, device, **kw)
            except Exception as e:
                print(f"  {name:26s} FAILED: {str(e)[:60]}")
                torch.cuda.empty_cache()
                continue
            if base is None:
                base = ms
            print(f"  {name:26s} {ms:7.1f} ms/step   {base/ms:4.2f}x")
        if args.profile:
            print(f"\n--- op breakdown ({task}, baseline) ---")
            profile_ops(config, task, device)
        print()


if __name__ == "__main__":
    main()
