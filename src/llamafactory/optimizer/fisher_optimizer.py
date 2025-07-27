import torch
from torch.optim import Optimizer

class FisherOptimizer(Optimizer):
    """Optimizer that accumulates the empirical diagonal Fisher information.

    This optimizer is *stateful* in the sense that for every parameter it
    stores:
        - fisher_sum: running sum of squared gradients
        - sample_count: number of optimizer.step() calls (can be interpreted
          as number of mini-batches processed)

    It performs **no** parameter update – model weights remain frozen.  The
    class is written to be fully compatible with DeepSpeed ZeRO stages: the
    state tensors are created on the same device as the parameter and will
    be partitioned / gathered by DeepSpeed exactly like any other optimizer
    state.
    """

    def __init__(self, params, *, lr: float = 0.0):
        # We keep a dummy lr hyper-parameter so that common Trainer code that
        # expects it does not break, but it is never used.
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """Accumulate squared gradients – do **not** modify parameters."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
                state = self.state[p]

                # Lazy state initialization (done on first gradient seen)
                if len(state) == 0:
                    state["fisher_sum"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Use a scalar tensor so it is easy for DeepSpeed to partition
                    state["sample_count"] = torch.zeros((), dtype=torch.float32, device=p.device)

                state["fisher_sum"].add_(grad.pow(2))
                state["sample_count"] += 1.0

        return loss 