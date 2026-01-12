from typing import List

import torch
import torch.nn as nn

from grouped_norm.ops import grouped_rmsnorm


class GroupedNorm(nn.Module):
    def __init__(
        self,
        num_norms: int,
        hidden_size: int,
        eps: float = 1e-5,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_norms = num_norms
        self.hidden_size = hidden_size
        self.eps = eps
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()

    def forward(self, x: torch.Tensor, m_splits: List[int]) -> torch.Tensor:
        raise NotImplementedError


class GroupedRMSNorm(GroupedNorm):
    def __init__(
        self,
        num_norms: int,
        hidden_size: int,
        eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(num_norms, hidden_size, eps, dtype)
        self.weight = nn.Parameter(torch.ones(num_norms, hidden_size, dtype=self.dtype))

    def forward(self, x: torch.Tensor, m_splits: List[int]) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (sum(m_splits), hidden_size)
            m_splits: List of length (num_norms) containing the number of tokens for each group.

        Returns:
            Normalized tensor of the same shape as input x.
        """
        return grouped_rmsnorm(x, self.weight, m_splits, self.eps)

    def extra_repr(self):
        return f"num_norms={self.num_norms}, hidden_size={self.hidden_size}, eps={self.eps}, dtype={self.dtype}"
