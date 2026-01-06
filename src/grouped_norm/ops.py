from typing import List

import torch
import triton
import triton.language as tl


@triton.jit
def _load_row(base, cols, stride, mask):
    return tl.load(base + cols * stride, mask, other=0.0).to(tl.float32)


@triton.jit
def _grouped_rmsnorm_fwd_kernel(
    X,  # in [B, H]
    Y,  # out [B, H]
    Gamma,  # in [G, H]
    Rstd,  # out [B]
    SegIndptr,  # in [G + 1], starting at 0, end at B
    stride_in_b: tl.constexpr,
    stride_in_h: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_h: tl.constexpr,
    stride_gamma_g: tl.constexpr,
    stride_gamma_h: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    group_id = tl.program_id(0)
    row_in_group = tl.program_id(1)
    row_start = tl.load(SegIndptr + group_id)
    row_end = tl.load(SegIndptr + group_id + 1)

    row_id = row_start + row_in_group
    if row_id >= row_end:
        return

    tl.assume(BLOCK_SIZE >= HIDDEN_SIZE)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < HIDDEN_SIZE

    x_base = X + row_id * stride_in_b
    y_base = Y + row_id * stride_out_b
    gamma_base = Gamma + group_id * stride_gamma_g

    x = _load_row(x_base, cols, stride_in_h, mask)
    gamma = _load_row(gamma_base, cols, stride_gamma_h, mask)

    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / HIDDEN_SIZE
    rstd = 1 / tl.sqrt(mean_sq + EPS)
    tl.store(Rstd + row_id, rstd)

    out = x * rstd * gamma
    tl.store(y_base + cols * stride_out_h, out, mask=mask)


@triton.jit
def _grouped_rmsnorm_bwd_dgamma_phase1(
    dGamma_partial,  # out [B, H]
    dY,  # in [B, H]
    X,  # in [B, H]
    Rstd,  # in [B]
    stride_dgammap_b: tl.constexpr,
    stride_dgammap_h: tl.constexpr,
    stride_dy_b: tl.constexpr,
    stride_dy_h: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_h: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tl.assume(BLOCK_SIZE >= HIDDEN_SIZE)

    row_id = tl.program_id(0)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < HIDDEN_SIZE

    out_base = dGamma_partial + row_id * stride_dgammap_b
    dy_base = dY + row_id * stride_dy_b
    x_base = X + row_id * stride_x_b

    dy = _load_row(dy_base, cols, stride_dy_h, mask)
    x = _load_row(x_base, cols, stride_x_h, mask)
    rstd = tl.load(Rstd + row_id)

    out = dy * x * rstd
    tl.store(out_base + cols * stride_dgammap_h, out, mask=mask)


@triton.jit
def _grouped_rmsnorm_bwd_dgamma_phase2(
    dGamma,  # out [G, H]
    dGamma_partial,  # in [B, H]
    SegIndptr,  # in [G + 1], starting at 0, end at B
    stride_dgamma_g: tl.constexpr,
    stride_dgamma_h: tl.constexpr,
    stride_dgammap_b: tl.constexpr,
    stride_dgammap_h: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    group_id = tl.program_id(0)
    col_id = tl.program_id(1)

    row_start = tl.load(SegIndptr + group_id)
    row_end = tl.load(SegIndptr + group_id + 1)

    blk_start = row_start // BLOCK_SIZE
    blk_end = tl.cdiv(row_end, BLOCK_SIZE)

    dgamma = 0.0
    for blk_id in tl.range(blk_start, blk_end):
        row_base = blk_id * BLOCK_SIZE
        rows = tl.arange(0, BLOCK_SIZE) + row_base
        mask = (rows >= row_start) & (rows < row_end)

        partial_ptrs = (
            dGamma_partial + rows * stride_dgammap_b + col_id * stride_dgammap_h
        )
        partial = tl.load(partial_ptrs, mask=mask, other=0.0).to(tl.float32)

        dgamma += tl.sum(partial, axis=0)

    dgamma_ptr = dGamma + group_id * stride_dgamma_g + col_id * stride_dgamma_h
    tl.store(dgamma_ptr, dgamma)


@triton.jit
def _grouped_rmsnorm_bwd_dx_kernel(
    dX,  # out [B, H]
    dY,  # in [B, H]
    X,  # in [B, H]
    Gamma,  # in [G, H]
    Rstd,  # in [B]
    SegIndptr,  # in [G + 1], starting at 0, end at B
    stride_dx_b: tl.constexpr,
    stride_dx_h: tl.constexpr,
    stride_dy_b: tl.constexpr,
    stride_dy_h: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_h: tl.constexpr,
    stride_gamma_g: tl.constexpr,
    stride_gamma_h: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tl.assume(BLOCK_SIZE >= HIDDEN_SIZE)

    group_id = tl.program_id(0)

    row_in_group = tl.program_id(1)
    row_start = tl.load(SegIndptr + group_id)
    row_end = tl.load(SegIndptr + group_id + 1)
    row_id = row_start + row_in_group
    if row_id >= row_end:
        return

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < HIDDEN_SIZE
    dx_base = dX + row_id * stride_dx_b
    dy_base = dY + row_id * stride_dy_b
    x_base = X + row_id * stride_x_b
    gamma_base = Gamma + group_id * stride_gamma_g

    dy = _load_row(dy_base, cols, stride_dy_h, mask)
    x = _load_row(x_base, cols, stride_x_h, mask)
    gamma = _load_row(gamma_base, cols, stride_gamma_h, mask)
    rstd = tl.load(Rstd + row_id)
    x_hat = x * rstd
    dy_hat = dy * gamma

    dx = rstd * (dy_hat - x_hat * tl.sum(dy_hat * x_hat, axis=0) / HIDDEN_SIZE)
    tl.store(dx_base + cols * stride_dx_h, dx, mask=mask)


@triton.jit
def _calculate_seg_indptr_kernel(
    SegIndptr,  # out [G + 1]
    MSplits,  # in [G]
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tl.assume(BLOCK_SIZE >= N)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    m_splits = tl.load(MSplits + offs, mask=mask, other=0)
    cumsum = tl.cumsum(m_splits, axis=0)

    tl.store(SegIndptr + 1 + offs, cumsum, mask=mask)
    tl.store(SegIndptr, 0)


def seg_indptr_from_splits(
    m_splits: torch.Tensor,
) -> torch.Tensor:
    g = m_splits.numel()
    device = m_splits.device

    seg_indptr = torch.empty(g + 1, dtype=torch.int32, device=device)

    BLOCK_SIZE = triton.next_power_of_2(g)
    _calculate_seg_indptr_kernel[(1,)](
        seg_indptr,
        m_splits,
        g,
        BLOCK_SIZE,
    )
    return seg_indptr


class GroupedRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        gamma: torch.Tensor,
        m_splits: List[int],
        eps: float,
    ):
        device = x.device
        b, h = x.shape
        g = gamma.shape[0]

        m_splits_dev = torch.tensor(m_splits, dtype=torch.int32, device=device)
        seg_indptr = seg_indptr_from_splits(m_splits_dev)

        max_split = max(m_splits)

        y = torch.empty_like(x)
        rstd = torch.empty(b, dtype=x.dtype, device=device)

        BLOCK_SIZE = triton.next_power_of_2(h)
        num_warps = max(4, min(16, BLOCK_SIZE // 256))

        _grouped_rmsnorm_fwd_kernel[(g, max_split)](
            x,
            y,
            gamma,
            rstd,
            seg_indptr,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            gamma.stride(0),
            gamma.stride(1),
            h,
            eps,
            BLOCK_SIZE,
            num_warps=num_warps,
        )

        ctx.save_for_backward(x, gamma, rstd, seg_indptr)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.max_split = max_split
        ctx.num_warps = num_warps

        return y

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, dy: torch.Tensor):
        x, gamma, rstd, seg_indptr = ctx.saved_tensors
        BLOCK_SIZE = ctx.BLOCK_SIZE
        max_split = ctx.max_split
        num_warps = ctx.num_warps
        b, h = x.shape
        g = gamma.shape[0]

        dx = torch.empty_like(x)
        dgamma_partial = torch.empty_like(x)
        dgamma = torch.empty_like(gamma)

        _grouped_rmsnorm_bwd_dx_kernel[(g, max_split)](
            dx,
            dy,
            x,
            gamma,
            rstd,
            seg_indptr,
            dx.stride(0),
            dx.stride(1),
            dy.stride(0),
            dy.stride(1),
            x.stride(0),
            x.stride(1),
            gamma.stride(0),
            gamma.stride(1),
            h,
            BLOCK_SIZE,
            num_warps=num_warps,
        )

        _grouped_rmsnorm_bwd_dgamma_phase1[(b,)](
            dgamma_partial,
            dy,
            x,
            rstd,
            dgamma_partial.stride(0),
            dgamma_partial.stride(1),
            dy.stride(0),
            dy.stride(1),
            x.stride(0),
            x.stride(1),
            h,
            BLOCK_SIZE,
            num_warps=num_warps,
        )

        PHASE2_BLOCK_SIZE = triton.next_power_of_2(max_split)
        phase2_num_warps = max(4, min(16, PHASE2_BLOCK_SIZE // 256))
        _grouped_rmsnorm_bwd_dgamma_phase2[(g, h)](
            dgamma,
            dgamma_partial,
            seg_indptr,
            dgamma.stride(0),
            dgamma.stride(1),
            dgamma_partial.stride(0),
            dgamma_partial.stride(1),
            PHASE2_BLOCK_SIZE,
            num_warps=phase2_num_warps,
        )

        return dx, dgamma, None, None


def grouped_rmsnorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    m_splits: List[int],
    eps: float = 1e-5,
) -> torch.Tensor:
    return GroupedRMSNormFunction.apply(x, gamma, m_splits, eps)
