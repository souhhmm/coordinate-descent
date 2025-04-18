from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT


class RandomizedCoordinateDescent(Optimizer):
    def __init__(self, params: ParamsT, lr: Union[float, Tensor] = 1e-3) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr)
        super(RandomizedCoordinateDescent, self).__init__(params, defaults)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super(RandomizedCoordinateDescent, self).__setstate__(state)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.numel() == 0:
                    continue

                # choose a random index
                idx = torch.randint(0, grad.numel(), (1,)).item()

                # update at that index
                p.data.view(-1)[idx] -= lr * grad.view(-1)[idx]
        return loss
