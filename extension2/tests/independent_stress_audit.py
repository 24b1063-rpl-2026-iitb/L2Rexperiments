import os
import sys
import time
import math
import subprocess
import torch
import torch.nn as nn
import numpy as np

# Ensure local extension2 package is on path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT2_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if EXT2_DIR not in sys.path:
    sys.path.insert(0, EXT2_DIR)

from allrank.models.feature_attention import FeatureTokenizer, LatentCrossAttention, IntraFeatureAttention
from allrank.models.model import FCModel


def test_extreme_inputs():
    print("=" * 60)
    print("AUDIT 2.a: EXTREME FEATURE INPUT STRESS TEST")
    print("=" * 60)

    n_features = 136
    d_model = 144
    b, s = 4, 30

    test_scenarios = [
        ("All Zeros (x = 0)", torch.zeros(b, s, n_features)),
        ("All Ones (x = 1)", torch.ones(b, s, n_features)),
        ("Extreme Large Positive (x = +10,000.0)", torch.full((b, s, n_features), 1e4)),
        ("Extreme Large Negative (x = -10,000.0)", torch.full((b, s, n_features), -1e4)),
        ("Uniform [-10,000.0, +10,000.0]", torch.empty(b, s, n_features).uniform_(-1e4, 1e4)),
        ("Near-Zero / Subnormal (x = 1e-7)", torch.full((b, s, n_features), 1e-7)),
        ("Mixed Normal with Extreme Outliers (+-1e4)", torch.randn(b, s, n_features)),
    ]

    # Add extreme outliers to scenario 7
    mask_outliers = torch.rand(b, s, n_features) < 0.05
    test_scenarios[-1][1][mask_outliers] = 1e4
    mask_neg = torch.rand(b, s, n_features) < 0.05
    test_scenarios[-1][1][mask_neg] = -1e4

    all_passed = True
    for name, tensor in test_scenarios:
        module = IntraFeatureAttention(n_features=n_features, d_model=d_model, n_heads=4, d_ff=288, dropout=0.0)
        module.train()

        x = tensor.clone().requires_grad_(True)
        out = module(x)

        # Check forward pass
        has_nan_fwd = torch.isnan(out).any().item()
        has_inf_fwd = torch.isinf(out).any().item()

        # Check backward pass
        loss = out.sum()
        loss.backward()

        has_nan_x_grad = torch.isnan(x.grad).any().item()
        has_inf_x_grad = torch.isinf(x.grad).any().item()

        param_grads_ok = True
        for pname, p in module.named_parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any().item() or torch.isinf(p.grad).any().item():
                    param_grads_ok = False
                    break

        status = (not has_nan_fwd) and (not has_inf_fwd) and (not has_nan_x_grad) and (not has_inf_x_grad) and param_grads_ok
        print(f"Scenario: {name:45s} | Fwd NaN/Inf: {has_nan_fwd}/{has_inf_fwd} | Back Grad OK: {param_grads_ok} | Passed: {status}")
        if not status:
            all_passed = False

    assert all_passed, "Extreme inputs stress test failed!"
    print(">>> 2.a PASS: All extreme inputs handled without NaN or Inf!\n")


def test_shape_edge_cases():
    print("=" * 60)
    print("AUDIT 2.b: SHAPE EDGE CASES STRESS TEST")
    print("=" * 60)

    n_features = 136
    d_model = 144

    test_shapes = [
        ("Minimal (B=1, S=1)", (1, 1, n_features)),
        ("Single Item Slate (B=16, S=1)", (16, 1, n_features)),
        ("Small Slate (B=2, S=5)", (2, 5, n_features)),
        ("Standard Slate (B=16, S=30)", (16, 30, n_features)),
        ("Full Paper Slate Length (B=8, S=240)", (8, 240, n_features)),
        ("Massive Stress Shape (B=32, S=240)", (32, 240, n_features)),
    ]

    all_passed = True
    for name, shape in test_shapes:
        b, s, f = shape
        module = IntraFeatureAttention(n_features=f, d_model=d_model, n_heads=4, d_ff=288, dropout=0.0)
        module.train()

        x = torch.randn(b, s, f, requires_grad=True)
        t0 = time.perf_counter()
        out = module(x)
        fwd_time = (time.perf_counter() - t0) * 1000.0

        assert out.shape == (b, s, d_model), f"Expected shape {(b, s, d_model)}, got {out.shape}"
        assert not torch.isnan(out).any(), f"NaN in forward pass for shape {shape}"
        assert not torch.isinf(out).any(), f"Inf in forward pass for shape {shape}"

        t0 = time.perf_counter()
        loss = out.sum()
        loss.backward()
        bwd_time = (time.perf_counter() - t0) * 1000.0

        assert x.grad is not None and not torch.isnan(x.grad).any(), f"NaN in gradient for shape {shape}"

        total_docs = b * s
        print(f"Shape: {name:32s} (Docs: {total_docs:5d}) | Fwd: {fwd_time:6.2f} ms | Bwd: {bwd_time:6.2f} ms | OK")

    print(">>> 2.b PASS: All shape edge cases (including B=1,S=1 and B=32,S=240) verified!\n")


def test_memory_and_latency_profiling():
    print("=" * 60)
    print("AUDIT 2.c: MEMORY FOOTPRINT & LATENCY PROFILING")
    print("=" * 60)

    n_features = 136
    d_model = 144

    # Baseline FCModel
    fc_model = FCModel(sizes=[d_model], input_norm=False, activation="ReLU", dropout=0.1, n_features=n_features)
    # Extension 2 IntraFeatureAttention
    ifa_model = IntraFeatureAttention(n_features=n_features, d_model=d_model, n_heads=4, d_ff=288, dropout=0.1)

    # 1. Parameter comparison
    fc_params = sum(p.numel() for p in fc_model.parameters())
    fc_bytes = sum(p.numel() * p.element_size() for p in fc_model.parameters())
    ifa_params = sum(p.numel() for p in ifa_model.parameters())
    ifa_bytes = sum(p.numel() * p.element_size() for p in ifa_model.parameters())

    print(f"Parameter Counts:")
    print(f"  FCModel:                 {fc_params:8,d} params ({fc_bytes / 1024:.2f} KB)")
    print(f"  IntraFeatureAttention:   {ifa_params:8,d} params ({ifa_bytes / 1024:.2f} KB)")
    print(f"  Delta:                  +{ifa_params - fc_params:,d} params (due to continuous tokenizer & cross-attention)")

    # 2. Latency Benchmarking (CPU / standard eval mode)
    benchmarks = [
        ("Typical Slate (B=16, S=30)", (16, 30, n_features), 100),
        ("Max Paper Slate (B=32, S=240)", (32, 240, n_features), 20),
    ]

    for bname, shape, runs in benchmarks:
        b, s, f = shape
        x = torch.randn(b, s, f)

        # Warmup
        fc_model.eval()
        ifa_model.eval()
        with torch.no_grad():
            for _ in range(5):
                _ = fc_model(x)
                _ = ifa_model(x)

        # FCModel forward
        t_fc_fwd = []
        with torch.no_grad():
            for _ in range(runs):
                t0 = time.perf_counter()
                _ = fc_model(x)
                t_fc_fwd.append((time.perf_counter() - t0) * 1000.0)

        # IntraFeatureAttention forward
        t_ifa_fwd = []
        with torch.no_grad():
            for _ in range(runs):
                t0 = time.perf_counter()
                _ = ifa_model(x)
                t_ifa_fwd.append((time.perf_counter() - t0) * 1000.0)

        mean_fc = np.mean(t_fc_fwd)
        std_fc = np.std(t_fc_fwd)
        mean_ifa = np.mean(t_ifa_fwd)
        std_ifa = np.std(t_ifa_fwd)

        print(f"\nLatency Benchmark [{bname} across {runs} iterations]:")
        print(f"  FCModel Forward:                {mean_fc:6.2f} +- {std_fc:5.2f} ms")
        print(f"  IntraFeatureAttention Forward:  {mean_ifa:6.2f} +- {std_ifa:5.2f} ms")
        print(f"  Ratio (IFA / FC):               {mean_ifa / mean_fc:6.2f}x")

    # 3. Training throughput (forward + backward)
    b, s, f = 16, 30, n_features
    fc_model.train()
    ifa_model.train()

    t_fc_train = []
    for _ in range(30):
        x = torch.randn(b, s, f, requires_grad=True)
        t0 = time.perf_counter()
        out = fc_model(x)
        out.sum().backward()
        t_fc_train.append((time.perf_counter() - t0) * 1000.0)

    t_ifa_train = []
    for _ in range(30):
        x = torch.randn(b, s, f, requires_grad=True)
        t0 = time.perf_counter()
        out = ifa_model(x)
        out.sum().backward()
        t_ifa_train.append((time.perf_counter() - t0) * 1000.0)

    print(f"\nTrain Step (Fwd + Bwd, B=16, S=30, 30 iterations):")
    print(f"  FCModel Train Step:             {np.mean(t_fc_train):6.2f} +- {np.std(t_fc_train):5.2f} ms")
    print(f"  IntraFeatureAttention Step:     {np.mean(t_ifa_train):6.2f} +- {np.std(t_ifa_train):5.2f} ms")
    print(">>> 2.c PASS: Memory footprint and latency profiles successfully collected!\n")


def test_extension1_untouched():
    print("=" * 60)
    print("AUDIT 2.d: INTEGRITY OF EXTENSION 1 REPOSITORY")
    print("=" * 60)

    repo_root = os.path.abspath(os.path.join(EXT2_DIR, ".."))
    ext1_dir = os.path.join(repo_root, "extension1")

    assert os.path.isdir(ext1_dir), f"Directory {ext1_dir} does not exist!"

    # Check git status for extension1
    res_status = subprocess.run(
        ["git", "status", "--porcelain", "extension1"],
        cwd=repo_root,
        capture_output=True,
        text=True
    )
    assert res_status.returncode == 0, f"git status failed: {res_status.stderr}"
    status_output = res_status.stdout.strip()

    # Check git diff for extension1
    res_diff = subprocess.run(
        ["git", "diff", "HEAD", "--", "extension1"],
        cwd=repo_root,
        capture_output=True,
        text=True
    )
    assert res_diff.returncode == 0, f"git diff failed: {res_diff.stderr}"
    diff_output = res_diff.stdout.strip()

    print(f"Git status porcelain on extension1: '{status_output}'")
    print(f"Git diff HEAD on extension1:        '{diff_output}'")

    if status_output == "" and diff_output == "":
        print(">>> 2.d PASS: extension1 is 100% INTACT, UNTOUCHED, and BIT-IDENTICAL to git HEAD!\n")
        return True
    else:
        print(">>> 2.d FAIL: extension1 has modifications!")
        return False


if __name__ == "__main__":
    test_extreme_inputs()
    test_shape_edge_cases()
    test_memory_and_latency_profiling()
    ext1_ok = test_extension1_untouched()
    if not ext1_ok:
        sys.exit(1)
    print("=" * 60)
    print("ALL INDEPENDENT VERIFICATION AUDITS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
