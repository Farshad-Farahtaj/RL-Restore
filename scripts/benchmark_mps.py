"""Measure real training throughput on this machine -> reports/benchmark.json.

The full.yaml budget MUST be justified by these numbers (spec design rule 3).
Usage: uv run python scripts/benchmark_mps.py [--quick]
"""

import argparse
import json
import time
from pathlib import Path

import torch

from rlrestore.tools.models import LargeToolCNN, SmallToolCNN


def time_train_steps(model, device, batch=64, steps=50, warmup=10):
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.rand(batch, 3, 63, 63, device=device)
    y = torch.rand(batch, 3, 63, 63, device=device)
    for _ in range(warmup):
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        opt.step()
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        opt.step()
    if device.type == "mps":
        torch.mps.synchronize()
    return steps / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny run for tests/CI")
    args = ap.parse_args()
    steps, warmup, batch = (3, 1, 8) if args.quick else (50, 10, 64)

    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available() and not args.quick:
        devices.insert(0, torch.device("mps"))

    results = {}
    for dev in devices:
        for name, cls in [("small_tool", SmallToolCNN), ("large_tool", LargeToolCNN)]:
            its = time_train_steps(cls(), dev, batch=batch, steps=steps, warmup=warmup)
            results[f"{name}@{dev.type}"] = round(its, 2)
            print(f"{name:>11} on {dev.type:>3}: {its:7.2f} it/s (batch {batch})")

    if not args.quick:
        full_iters = 20 * 4000  # full.yaml epochs * steps_per_epoch
        for name in ("small_tool", "large_tool"):
            key = f"{name}@mps" if f"{name}@mps" in results else f"{name}@cpu"
            hours = full_iters / results[key] / 3600
            results[f"{name}_full_run_hours"] = round(hours, 2)
            print(f"{name}: full.yaml run = {hours:.1f} h per tool")

    if args.quick:
        print("(quick mode: skipping reports/benchmark.json write)")
        return
    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(results, indent=2) + "\n")
    print("wrote reports/benchmark.json")


if __name__ == "__main__":
    main()
