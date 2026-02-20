from __future__ import annotations

from collections.abc import Iterable

import torch

from classes.training import TrainParameters


def build_adamw_with_optional_cosine_scheduler(
    parameters: Iterable[torch.nn.Parameter],
    train_parameters: TrainParameters,
    total_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    optimizer = torch.optim.AdamW(
        params=list(parameters),
        **train_parameters.optimizer_parameters,
    )

    if train_parameters.scheduler_parameters:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_steps,
            **train_parameters.scheduler_parameters,
        )
    else:
        scheduler = None

    return optimizer, scheduler
