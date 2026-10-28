"""Controlled allocation-initialization ablation for exclusive worker processes.

Only PyTorch's fill of *uninitialized* allocations changes. Deterministic
algorithms and real zeros/ones remain enabled and unchanged. The flag is
process-global: this context is for an exclusively owned benchmark worker,
not a concurrent serving API or a thread-local execution option.
"""
from contextlib import contextmanager

import torch
import torch.utils.deterministic


@contextmanager
def allocation_policy(fill: bool):
    """Temporarily select empty-allocation fills and yield a JSON policy record.

    Callers must already have deterministic algorithms enabled and exclusively
    own all PyTorch work in this process throughout the context. Nesting in the
    same execution thread restores the enclosing policy; exceptions restore the
    previous flag too. No model, arithmetic, packet or explicit fill is changed.
    """
    if type(fill) is not bool:
        raise TypeError('fill must be a bool')
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError('deterministic algorithms must already be enabled')
    previous = torch.utils.deterministic.fill_uninitialized_memory
    torch.utils.deterministic.fill_uninitialized_memory = fill
    try:
        yield dict(
            fill_uninitialized_memory=torch.utils.deterministic.fill_uninitialized_memory,
            previous_fill_uninitialized_memory=previous,
            deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
            deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            scope='process-global flag; exclusively owned benchmark worker; explicit zeros/ones unchanged')
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous
