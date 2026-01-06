import random
from typing import List

import torch
import torch.nn as nn

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
        self.weight = nn.Parameter(torch.ones(num_norms, hidden_size))

    def forward(self, x: torch.Tensor, m_splits: List[int]) -> torch.Tensor:
        outputs = []
        normalized_shape = (self.hidden_size,)

        start = 0
        for i, split in enumerate(m_splits):
            end = start + split

            x_split = x[start:end, :]
            gamma = self.weight[i].flatten()
            outputs.append(torch.rms_norm(x_split, normalized_shape, gamma, self.eps))

            start = end

        return torch.cat(outputs, dim=0)


def test_grouped_rmsnorm_equivalence():
    torch.manual_seed(0)
    random.seed(0)

    num_norms = 32
    hidden_size = 512
    eps = 1e-5

    torch.set_default_device(torch.device("cuda:0"))
    torch.set_default_dtype(torch.float32)

    grouped_rmsnorm = GroupedRMSNorm(num_norms, hidden_size, eps).cuda()
    sequential_rmsnorm = SequentialRMSNorm(num_norms, hidden_size, eps).cuda()

    m_splits = [random.randint(0, 96) for _ in range(num_norms)]
    batch_size = sum(m_splits)

    x_grouped = torch.randn(
        batch_size,
        hidden_size,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    x_sequential = x_grouped.clone().detach().requires_grad_(True)

    out_grouped = grouped_rmsnorm(x_grouped, m_splits)
    out_sequential = sequential_rmsnorm(x_sequential, m_splits)

    torch.testing.assert_close(out_grouped, out_sequential, rtol=1e-5, atol=1e-5)

    grad_output = torch.randn_like(out_grouped)
    out_grouped.backward(grad_output)
    out_sequential.backward(grad_output)

    torch.testing.assert_close(x_grouped.grad, x_sequential.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        grouped_rmsnorm.weight.grad,
        sequential_rmsnorm.weight.grad,
        rtol=1e-5,
        atol=1e-5,
    )
