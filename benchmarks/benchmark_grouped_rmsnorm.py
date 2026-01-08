from typing import Literal
import os
from functools import partial
from typing import List

import torch
import triton
import triton.testing

from grouped_norm.ops import grouped_rmsnorm


def sequential_rmsnorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    m_splits: List[int],
    eps: float = 1e-5,
):
    outputs = []
    start = 0
    normalized_shape = (x.shape[1],)
    for i, count in enumerate(m_splits):
        end = start + count
        if count > 0:
            x_split = x[start:end]
            gamma_group = gamma[i].flatten()
            output = torch.rms_norm(x_split, normalized_shape, gamma_group, eps)
            outputs.append(output)
        start = end
    return torch.cat(outputs, dim=0)


def make_input(
    N: int,
    H: int,
    num_groups: int,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cuda",
):
    # Simple split generation: roughly equal splits
    avg_split = N // num_groups
    m_splits = [avg_split] * num_groups
    m_splits[-1] += N - sum(m_splits)

    x = torch.randn(N, H, dtype=dtype, device=device, requires_grad=True)
    gamma = torch.randn(num_groups, H, dtype=dtype, device=device, requires_grad=True)
    gamma_global = torch.randn(H, dtype=dtype, device=device, requires_grad=True)

    return x, gamma, gamma_global, m_splits


def benchmark_impl(
    N: int,
    H: int,
    num_groups: int,
    provider: Literal["torch", "triton", "global"],
    mode: Literal["fwd", "bwd"],
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cuda",
):
    torch.manual_seed(0)

    # Simple split generation: roughly equal splits
    x, gamma, gamma_global, m_splits = make_input(N, H, num_groups, dtype, device)
    eps = 1e-5
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        fn = partial(sequential_rmsnorm, x, gamma, m_splits, eps)
    elif provider == "triton":
        fn = partial(grouped_rmsnorm, x, gamma, m_splits, eps)
    elif provider == "global":
        # Speed of light baseline: Standard fused RMSNorm on the whole tensor
        # ignoring groups. This isn't mathematically equivalent but shows peak BW.
        fn = partial(torch.nn.functional.rms_norm, x, (H,), gamma_global, eps)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if mode == "bwd":
        y = fn()
        dy = torch.randn_like(y)

        def bwd_fn():
            y.backward(dy, retain_graph=True)
            if x.grad is not None:
                x.grad = None
            if provider == "global":
                if gamma_global.grad is not None:
                    gamma_global.grad = None
            else:
                if gamma.grad is not None:
                    gamma.grad = None

        ms, min_ms, max_ms = triton.testing.do_bench(bwd_fn, quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    # GB/s calculation
    element_size = x.element_size()
    if mode == "fwd":
        # Read X, Gamma. Write Y.
        if provider == "global":
            total_bytes = (2 * N * H + H) * element_size
        else:
            # X: N*H, Gamma: G*H, Y: N*H
            total_bytes = (2 * N * H + num_groups * H) * element_size
    else:
        # Backward:
        # Read: dY (N*H), X (N*H), Gamma (G*H or H), Rstd (N)
        # Write: dX (N*H), dGamma (G*H or H)
        if provider == "global":
            total_bytes = (3 * N * H + 2 * H + N) * element_size
        else:
            total_bytes = (3 * N * H + 2 * num_groups * H + N) * element_size

    return total_bytes * 1e-9 / (ms * 1e-3)


COMMON_BENCHMARK_CFG = {
    "x_names": ["N"],  # Total number of tokens
    "x_vals": [1024 * i for i in range(1, 21)],
    "line_arg": "provider",
    "line_vals": ["triton", "torch", "global"],
    "line_names": ["Triton", "Torch", "Global RMSNorm"],
    "styles": [("blue", "-"), ("green", "-"), ("red", "--")],
    "ylabel": "GB/s",
}
COMMON_ARGS = {
    "H": 4096,
    "num_groups": 32,
}


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            **COMMON_BENCHMARK_CFG,
            plot_name="grouped-rmsnorm-fwd-float32",
            args={**COMMON_ARGS, "mode": "fwd", "dtype": torch.float32},
        ),
        triton.testing.Benchmark(
            **COMMON_BENCHMARK_CFG,
            plot_name="grouped-rmsnorm-fwd-float16",
            args={**COMMON_ARGS, "mode": "fwd", "dtype": torch.float16},
        ),
        triton.testing.Benchmark(
            **COMMON_BENCHMARK_CFG,
            plot_name="grouped-rmsnorm-fwd-bfloat16",
            args={**COMMON_ARGS, "mode": "fwd", "dtype": torch.bfloat16},
        ),
    ]
)
def benchmark_fwd(
    N: int,
    H: int,
    num_groups: int,
    provider: Literal["triton", "torch", "global"],
    mode: Literal["fwd", "bwd"],
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cuda",
):
    return benchmark_impl(N, H, num_groups, provider, mode, dtype, device)


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            **COMMON_BENCHMARK_CFG,
            plot_name="grouped-rmsnorm-bwd-float32",
            args={**COMMON_ARGS, "mode": "bwd", "dtype": torch.float32},
        ),
        triton.testing.Benchmark(
            **COMMON_BENCHMARK_CFG,
            plot_name="grouped-rmsnorm-bwd-float16",
            args={**COMMON_ARGS, "mode": "bwd", "dtype": torch.float16},
        ),
        triton.testing.Benchmark(
            **COMMON_BENCHMARK_CFG,
            plot_name="grouped-rmsnorm-bwd-bfloat16",
            args={**COMMON_ARGS, "mode": "bwd", "dtype": torch.bfloat16},
        ),
    ]
)
def benchmark_bwd(
    N: int,
    H: int,
    num_groups: int,
    provider: Literal["triton", "torch", "global"],
    mode: Literal["fwd", "bwd"],
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cuda",
):
    return benchmark_impl(N, H, num_groups, provider, mode, dtype, device)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    benchmark_fwd.run(save_path="results", print_data=True)
    benchmark_bwd.run(save_path="results", print_data=True)
