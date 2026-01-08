from typing import TypeVar
from typing import TypedDict
from dataclasses import dataclass
import random
from typing import List

import torch
import torch.nn as nn
import pytest

from grouped_norm import GroupedNorm, GroupedRMSNorm


class SequentialRMSNorm(GroupedNorm):
    def __init__(
        self,
        num_norms: int,
        hidden_size: int,
        eps: float = 1e-5,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(num_norms, hidden_size, eps, dtype)
        self.weight = nn.Parameter(torch.ones(num_norms, hidden_size, dtype=self.dtype))

    def forward(self, x: torch.Tensor, m_splits: List[int]) -> torch.Tensor:
        outputs = []
        normalized_shape = (self.hidden_size,)

        start = 0
        for i, split in enumerate(m_splits):
            end = start + split

            if split > 0:
                x_split = x[start:end, :]
                gamma = self.weight[i].flatten()
                outputs.append(
                    torch.rms_norm(x_split, normalized_shape, gamma, self.eps)
                )

            start = end

        return torch.cat(outputs, dim=0)


def setup_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)


class AllocKwargs(TypedDict):
    device: torch.device
    dtype: torch.dtype


class TolKwargs(TypedDict):
    rtol: float
    atol: float


NormT = TypeVar("NormT", bound=GroupedNorm)


@dataclass
class ModelConfig:
    num_norms: int
    hidden_size: int
    eps: float

    dtype: torch.dtype
    device: torch.device

    @property
    def alloc_kwargs(self):
        return AllocKwargs(device=self.device, dtype=self.dtype)

    @property
    def tolerance(self):
        tol = 1e-2 if self.dtype.itemsize <= 2 else 1e-5
        return TolKwargs(rtol=tol, atol=tol)

    def build_model(self, cls: type[NormT]) -> NormT:
        return cls(
            self.num_norms,
            self.hidden_size,
            self.eps,
            self.dtype,
        ).to(device=self.device)


DEVICE = torch.device("cuda:0")


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("target_batch_size", [2048, 4096])
def test_grouped_rmsnorm_equivalence_randomized(
    seed: int,
    dtype: torch.dtype,
    target_batch_size: int,
):
    setup_seed(seed)
    config = ModelConfig(
        num_norms=32,
        hidden_size=512,
        eps=1e-5,
        dtype=dtype,
        device=DEVICE,
    )

    impl = config.build_model(GroupedRMSNorm)
    ref_impl = config.build_model(SequentialRMSNorm)

    weights = torch.randn(
        config.num_norms,
        config.hidden_size,
        **config.alloc_kwargs,
    )
    impl.weight.data.copy_(weights)
    ref_impl.weight.data.copy_(weights)

    avg_bs = target_batch_size // config.num_norms
    m_splits = [random.randint(0, avg_bs * 2) for _ in range(config.num_norms)]
    batch_size = sum(m_splits)

    x_impl = torch.randn(
        batch_size,
        config.hidden_size,
        requires_grad=True,
        **config.alloc_kwargs,
    )
    x_ref = x_impl.clone().detach().requires_grad_(True)

    y_impl = impl(x_impl, m_splits)
    y_ref = ref_impl(x_ref, m_splits)
    torch.testing.assert_close(y_impl, y_ref, **config.tolerance)

    grad_output = torch.randn_like(y_impl)
    y_impl.backward(grad_output)
    y_ref.backward(grad_output)
    torch.testing.assert_close(x_impl.grad, x_ref.grad, **config.tolerance)
    torch.testing.assert_close(
        impl.weight.grad, ref_impl.weight.grad, **config.tolerance
    )
