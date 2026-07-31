import copy
import time
import numpy as np
import torch
from src.utils import get_device
from src.utils import load_model


def benchmark(
    model,
    x_single,
    n_runs,
    infer_batch_size,
    batch_size,
    label,
    dtype=torch.float32,
):
    device = next(model.parameters()).device

    # Pre-allocate batches
    batches = []
    for i in range(0, batch_size, infer_batch_size):
        current_bs = min(infer_batch_size, batch_size - i)
        batches.append(x_single.repeat(current_bs, 1).to(device=device, dtype=dtype))

    # Warm-up
    warmup_x = x_single.repeat(infer_batch_size, 1).to(
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        for _ in range(5):
            model(warmup_x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []

    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            for x_batch in batches:
                model(x_batch)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            times.append((end - start) * 1000)

    print(f"\n{label}")
    print(f"  Mean : {np.mean(times):.2f} ms")
    print(f"  Min  : {np.min(times):.2f} ms")


def test_inference(config):

    n_runs = config["testing"]["N_RUNS"]
    infer_batch_size = config["testing"]["INFER_BATCH_SIZE"]
    batch_size = config["testing"]["BATCH_SIZE"]

    device = get_device(config)

    model, _ = load_model(config, device)
    model.eval()

    T1 = np.log(0.8)
    T2 = np.log(0.07)
    B1 = 1.0

    x_single = torch.tensor(
        [[T1, T2, B1]],
        dtype=torch.float32,
        device=device,
    )

    # --------------------------------------------------
    # Baseline float32
    # --------------------------------------------------
    print("\n=== Baseline (float32) ===")

    benchmark(
        model,
        x_single,
        n_runs,
        infer_batch_size,
        batch_size,
        "float32",
        dtype=torch.float32,
    )

    # --------------------------------------------------
    # Float16
    # --------------------------------------------------
    print("\n=== Half precision (float16) ===")

    model_fp16 = copy.deepcopy(model).half().eval()

    benchmark(
        model_fp16,
        x_single,
        n_runs,
        infer_batch_size,
        batch_size,
        "float16",
        dtype=torch.float16,
    )

    # --------------------------------------------------
    # torch.compile
    # --------------------------------------------------
    print("\n=== torch.compile ===")

    model_compiled = torch.compile(copy.deepcopy(model))

    benchmark(
        model_compiled,
        x_single,
        n_runs,
        infer_batch_size,
        batch_size,
        "torch.compile",
        dtype=torch.float32,
    )

    # --------------------------------------------------
    # torch.compile + float16
    # --------------------------------------------------
    print("\n=== torch.compile + float16 ===")

    model_compiled_fp16 = torch.compile(copy.deepcopy(model).half())

    benchmark(
        model_compiled_fp16,
        x_single,
        n_runs,
        infer_batch_size,
        batch_size,
        "torch.compile + float16",
        dtype=torch.float16,
    )

    # --------------------------------------------------
    # Compare float32 vs float16 predictions
    # --------------------------------------------------
    with torch.no_grad():
        pred_fp32 = model(x_single.float())

        pred_fp16 = model_fp16(x_single.half())

    sig_fp32 = pred_fp32[0].cpu().numpy()
    sig_fp16 = pred_fp16[0].float().cpu().numpy()

    diff = np.abs(sig_fp32 - sig_fp16)

    print("\n=== Prediction comparison (float32 vs float16) ===")
    print(f"Max difference:  {diff.max():.6f}")
    print(f"Mean difference: {diff.mean():.6f}")
    print(f"Relative error: {(diff / (np.abs(sig_fp32) + 1e-12)).mean() * 100:.4f}%")
