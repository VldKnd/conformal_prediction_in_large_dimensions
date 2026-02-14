from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Self, Literal, Optional

import torch
import torch.nn as nn
from tqdm import trange

from classes.protocol import PushForwardOperator
from classes.training import TrainParameters
from pushforward_operators.picnn import FFNN



Direction = Literal["forward", "backward"]


@dataclass
class IPFParameters:
    outer_iterations: int = 10              
    inner_steps: int = 2000                
    num_sde_steps: int = 100                
    sigma: float = 0.5
    eps_time: float = 1e-5                  
    first_coupling: Literal["independent"] = "independent"


class SchrodingerBridgeQuantile(nn.Module, PushForwardOperator):
    def __init__(
        self,
        feature_dimension: int,
        response_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
    ):
        super().__init__()
        self.init_dict = {
            "feature_dimension": feature_dimension,
            "response_dimension": response_dimension,
            "hidden_dimension": hidden_dimension,
            "number_of_hidden_layers": number_of_hidden_layers,
        }
        self.response_dimension = response_dimension
        self.model_information_dict = {"class_name": "SchrodingerBridgeQuantile"}

        self.time_embedding_network = nn.Sequential(
            nn.Linear(1, feature_dimension),
            nn.Tanh(),
        )

        self.forward_drift = FFNN(
            feature_dimension=2 * feature_dimension,
            response_dimension=response_dimension,
            hidden_dimension=hidden_dimension,
            number_of_hidden_layers=number_of_hidden_layers,
            output_dimension=response_dimension,
        )
        self.backward_drift = FFNN(
            feature_dimension=2 * feature_dimension,
            response_dimension=response_dimension,
            hidden_dimension=hidden_dimension,
            number_of_hidden_layers=number_of_hidden_layers,
            output_dimension=response_dimension,
        )

        self.Y_scaler = nn.BatchNorm1d(response_dimension, affine=False)

    def warmup_y_scaler(self, dataloader: torch.utils.data.DataLoader) -> None:
        self.Y_scaler.train()
        with torch.no_grad():
            for _, y_batch in dataloader:
                _ = self.Y_scaler(y_batch)
        self.Y_scaler.eval()

    def scale_y(self, y_tensor: torch.Tensor) -> torch.Tensor:
        return self.Y_scaler(y_tensor)

    def unscale_y(self, y_scaled: torch.Tensor) -> torch.Tensor:
        return (
            y_scaled * torch.sqrt(self.Y_scaler.running_var + self.Y_scaler.eps)
            + self.Y_scaler.running_mean
        )

    def _predict_drift(
        self,
        direction: Direction,
        state: torch.Tensor,
        condition_x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        time_emb = self.time_embedding_network(t)
        cond = torch.cat([condition_x, time_emb], dim=-1)
        if direction == "forward":
            return self.forward_drift(condition=cond, tensor=state)
        elif direction == "backward":
            return self.backward_drift(condition=cond, tensor=state)
        else:
            raise ValueError(f"Unknown direction: {direction}")

    @torch.no_grad()
    def _get_train_tuple(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        x: torch.Tensor,
        direction: Direction,
        sigma: float,
        eps_time: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
          z_t = t z1 + (1-t) z0 + sigma * sqrt(t(1-t)) * eps
        Targets:
          forward: (z1-z0) - sigma * sqrt(t/(1-t)) * eps
          backward: -(z1-z0) - sigma * sqrt((1-t)/t) * eps
        """
        device = z0.device
        dtype = z0.dtype
        bsz = z0.shape[0]

        t = torch.rand((bsz, 1), device=device, dtype=dtype) * (1.0 - 2.0 * eps_time) + eps_time

        eps = torch.randn_like(z0)
        z_t = t * z1 + (1.0 - t) * z0
        z_t = z_t + sigma * torch.sqrt(t * (1.0 - t)) * eps

        delta = z1 - z0
        if direction == "forward":
            target = delta - sigma * torch.sqrt(t / (1.0 - t)) * eps
        elif direction == "backward":
            target = -delta - sigma * torch.sqrt((1.0 - t) / t) * eps
        else:
            raise ValueError(direction)

        return z_t, t, target, x
    
    @torch.no_grad()
    def _sample_sde(
        self,
        direction: Direction,
        z_start: torch.Tensor,
        x: torch.Tensor,
        num_steps: int,
        sigma: float,
    ) -> torch.Tensor:
        self.eval()
        z = z_start.detach().clone()
        dt = 1.0 / float(num_steps)
        bsz = z.shape[0]
        device, dtype = z.device, z.dtype

        for k in range(num_steps):
            tk = float(k) / float(num_steps)
            if direction == "backward":
                tk = 1.0 - tk
            t = torch.full((bsz, 1), tk, device=device, dtype=dtype)

            drift = self._predict_drift(direction=direction, state=z, condition_x=x, t=t)
            z = z + drift * dt
            z = z + sigma * torch.randn_like(z) * (dt ** 0.5)

        return z.detach()

    @torch.no_grad()
    def _generate_coupled_endpoints(
        self,
        x: torch.Tensor,
        y_scaled: torch.Tensor,
        prev_model: Optional["SchrodingerBridgeQuantile"],
        prev_direction: Optional[Direction],
        direction_to_train: Direction,
        ipf: IPFParameters,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device, dtype = y_scaled.device, y_scaled.dtype
        bsz = y_scaled.shape[0]

        u = torch.randn_like(y_scaled)

        if prev_model is None:
            perm = torch.randperm(bsz, device=device)
            y_perm = y_scaled[perm]

            if direction_to_train == "backward":
                # train y -> u
                z0, z1 = y_perm, u
            else:
                # train u -> y
                z0, z1 = u, y_perm
            return z0, z1
        
        assert prev_direction in ("forward", "backward")

        if direction_to_train == "backward":
            # Need pairs (y_like, u_like) where y_like is "start" and u_like is "end"
            if prev_direction == "forward":
                y_hat = prev_model._sample_sde(
                    direction="forward", z_start=u, x=x, num_steps=ipf.num_sde_steps, sigma=ipf.sigma
                )
                z0, z1 = y_hat, u
            else:
                u_hat = prev_model._sample_sde(
                    direction="backward", z_start=y_scaled, x=x, num_steps=ipf.num_sde_steps, sigma=ipf.sigma
                )
                z0, z1 = y_scaled, u_hat
            return z0, z1

        else:
            # direction_to_train == "fwd": want pairs (u_like, y_like)
            if prev_direction == "backward":
                # Sample u_hat = prev(y)
                u_hat = prev_model._sample_sde(
                    direction="backward", z_start=y_scaled, x=x, num_steps=ipf.num_sde_steps, sigma=ipf.sigma
                )
                z0, z1 = u_hat, y_scaled
            else:
                # If prev was fwd, sample y_hat from u, then pair (u, y_hat)
                y_hat = prev_model._sample_sde(
                    direction="forward", z_start=u, x=x, num_steps=ipf.num_sde_steps, sigma=ipf.sigma
                )
                z0, z1 = u, y_hat
            return z0, z1

    def make_progress_bar_message(
        self,
        direction: Direction,
        outer_it: int,
        step: int,
        losses: list[float],
        last_lr: Optional[float] = None,
    ) -> str:
        window = losses[-50:] if len(losses) >= 50 else losses
        running = sum(window) / max(1, len(window))
        msg = f"IPF {outer_it} [{direction}] step {step}, loss {running:.4f}"
        if last_lr is not None:
            msg += f", LR {last_lr:.6f}"
        return msg

    def fit(
        self,
        dataloader: torch.utils.data.DataLoader,
        train_parameters: TrainParameters,
        ipf_parameters: Optional[IPFParameters] = None,
        *args,
        **kwargs,
    ) -> Self:
        ipf = ipf_parameters or IPFParameters()

        # scaler warmup
        self.warmup_y_scaler(dataloader)
        self.Y_scaler.eval()

        # train mode
        self.forward_drift.train()
        self.backward_drift.train()
        self.time_embedding_network.train()

        # Separate optimizers is often cleaner for IPF
        opt_fwd = torch.optim.AdamW(
            list(self.forward_drift.parameters()) + list(self.time_embedding_network.parameters()),
            **train_parameters.optimizer_parameters,
        )
        opt_bwd = torch.optim.AdamW(
            list(self.backward_drift.parameters()) + list(self.time_embedding_network.parameters()),
            **train_parameters.optimizer_parameters,
        )

        sched_fwd = None
        sched_bwd = None
        if train_parameters.scheduler_parameters:
            # use total steps = outer * inner for each direction
            total_steps = ipf.outer_iterations * ipf.inner_steps
            sched_fwd = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=opt_fwd,
                T_max=total_steps,
                **train_parameters.scheduler_parameters,
            )
            sched_bwd = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=opt_bwd,
                T_max=total_steps,
                **train_parameters.scheduler_parameters,
            )

        verbose = train_parameters.verbose

        history = []
        prev_snapshot: Optional[SchrodingerBridgeQuantile] = None
        prev_direction: Optional[Direction] = None

        # Outer IPF loop
        for outer_it in range(1, ipf.outer_iterations + 1):
            # Alternate directions in the standard way: first train bwd, then fwd
            for direction in ("backward", "forward"):
                start_t = time.perf_counter()
                losses: list[float] = []

                # Select optimizer/scheduler
                if direction == "forward":
                    optimizer = opt_fwd
                    scheduler = sched_fwd
                else:
                    optimizer = opt_bwd
                    scheduler = sched_bwd

                # Inner steps: stream minibatches from dataloader (cycle if needed)
                dl_iter = iter(dataloader)

                progress = trange(
                    1,
                    ipf.inner_steps + 1,
                    desc=f"Training IPF {outer_it} [{direction}]",
                    disable=not verbose,
                )

                for step in progress:
                    try:
                        x_batch, y_batch = next(dl_iter)
                    except StopIteration:
                        dl_iter = iter(dataloader)
                        x_batch, y_batch = next(dl_iter)

                    y_scaled = self.scale_y(y_batch)

                    # Build endpoints (z0,z1) according to IPF coupling rule
                    z0, z1 = self._generate_coupled_endpoints(
                        x=x_batch,
                        y_scaled=y_scaled,
                        prev_model=prev_snapshot,
                        prev_direction=prev_direction,
                        direction_to_train=direction,
                        ipf=ipf,
                    )

                    # Training tuple
                    z_t, t, target, x_cond = self._get_train_tuple(
                        z0=z0,
                        z1=z1,
                        x=x_batch,
                        direction=direction,
                        sigma=ipf.sigma,
                        eps_time=ipf.eps_time,
                    )

                    # Predict drift
                    optimizer.zero_grad(set_to_none=True)
                    pred = self._predict_drift(direction=direction, state=z_t, condition_x=x_cond, t=t)

                    loss = (target - pred).view(pred.shape[0], -1).pow(2).sum(dim=1).mean()
                    if torch.isnan(loss).any():
                        raise ValueError("Loss is NaN")

                    loss.backward()

                    nn.utils.clip_grad_norm_(self.time_embedding_network.parameters(), max_norm=1.0)
                    if direction == "forward":
                        nn.utils.clip_grad_norm_(self.forward_drift.parameters(), max_norm=1.0)
                    else:
                        nn.utils.clip_grad_norm_(self.backward_drift.parameters(), max_norm=1.0)

                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                    loss_val = float(loss.item())
                    losses.append(loss_val)

                    if verbose:
                        last_lr = scheduler.get_last_lr()[0] if scheduler is not None else None
                        progress.set_description(
                            self.make_progress_bar_message(
                                direction=direction,  # type: ignore[arg-type]
                                outer_it=outer_it,
                                step=step,
                                losses=losses,
                                last_lr=last_lr,
                            )
                        )

                progress.close()

                history.append(
                    {
                        "outer_iteration": outer_it,
                        "direction": direction,
                        "mean_loss": float(torch.tensor(losses).mean().item()) if losses else float("nan"),
                        "training_time": time.perf_counter() - start_t,
                    }
                )
                
                prev_snapshot = self._make_frozen_copy(device=next(self.parameters()).device)
                prev_direction = direction  # type: ignore[assignment]

        self.model_information_dict.update(
            {
                "ipf_parameters": ipf.__dict__,
                "training_information": history,
                "training_batch_size": dataloader.batch_size,
            }
        )
        return self

    @torch.no_grad()
    def _make_frozen_copy(self, device: torch.device) -> "SchrodingerBridgeQuantile":
        copy_model = SchrodingerBridgeQuantile(**self.init_dict).to(device=device)
        copy_model.load_state_dict(self.state_dict())
        copy_model.Y_scaler.load_state_dict(self.Y_scaler.state_dict())
        copy_model.eval()
        for p in copy_model.parameters():
            p.requires_grad_(False)
        return copy_model

    @torch.no_grad()
    def push_u_given_x(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
        number_of_evaluations: int = 100
    ) -> torch.Tensor:
        self.eval()
        
        dt = torch.full(
            (u.shape[0], 1),
            1.0 / number_of_evaluations,
            device=u.device,
            dtype=u.dtype,
        )
        state = u

        for evaluation_index in range(number_of_evaluations):
            interpolation_times = (evaluation_index + 0.5) * dt
            vector_field_prediction = self._predict_drift(
                direction="forward",
                state=state,
                condition_x=x,
                t=interpolation_times,
            )
            state = state + dt * vector_field_prediction

        return self.unscale_y(state).detach()
    
    @torch.no_grad()
    def push_y_given_x(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        number_of_evaluations: int = 100
    ) -> torch.Tensor:
        self.eval()
        y_scaled = self.scale_y(y)
        dt = torch.full(
            (y_scaled.shape[0], 1),
            1.0 / number_of_evaluations,
            device=y_scaled.device,
            dtype=y_scaled.dtype,
        )
        state = y_scaled

        for evaluation_index in range(number_of_evaluations):
            interpolation_times = 1 - (evaluation_index + 0.5) * dt
            vector_field_prediction = self._predict_drift(
                direction="backward",
                state=state,
                condition_x=x,
                t=interpolation_times,
            )
            state = state + dt * vector_field_prediction

        return state.detach()
    

    def save(self, path: str) -> None:
        torch.save(
            {
                "init_dict": self.init_dict,
                "state_dict": self.state_dict(),
                "model_information_dict": self.model_information_dict,
            },
            path,
        )

    def load(self, path: str, map_location: torch.device = torch.device("cpu")) -> Self:
        data = torch.load(path, map_location=map_location)
        self.load_state_dict(data["state_dict"])
        self.model_information_dict = data.get("model_information_dict", {})
        return self

    @classmethod
    def load_class(cls, path: str, map_location: torch.device = torch.device("cpu")) -> Self:
        data = torch.load(path, map_location=map_location, weights_only=False)
        operator = cls(**data["init_dict"])
        operator.load_state_dict(data["state_dict"])
        operator.model_information_dict = data.get("model_information_dict", {})
        return operator