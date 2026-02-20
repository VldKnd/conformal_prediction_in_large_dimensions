from __future__ import annotations

import time

import torch
import torch.nn as nn
from tqdm import trange

from typing import Self
from classes.protocol import PushForwardOperator
from classes.training import TrainParameters
from pushforward_operators.picnn import FFNN


class FlowMatchingQuantile(nn.Module, PushForwardOperator):
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
        self.model_information_dict = {
            "class_name": "FlowMatchingQuantile",
        }

        self.time_embedding_network = nn.Sequential(
            nn.Linear(1, feature_dimension),
            nn.Tanh(),
        )
        self.vector_field_network = FFNN(
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

    def unscale_y(self, y_scaled: torch.Tensor) -> torch.Tensor:
        return (
            y_scaled * torch.sqrt(self.Y_scaler.running_var + self.Y_scaler.eps)
            + self.Y_scaler.running_mean
        )

    def scale_y(self, y_tensor: torch.Tensor) -> torch.Tensor:
        return self.Y_scaler(y_tensor)

    def predict_vector_field(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        interpolation_times: torch.Tensor,
    ) -> torch.Tensor:
        time_embedding = self.time_embedding_network(interpolation_times)
        return self.vector_field_network(
            condition=torch.cat([condition, time_embedding], dim=-1),
            tensor=state,
        )

    def make_progress_bar_message(
        self,
        training_information: list[dict],
        epoch_idx: int,
        last_learning_rate: float | None = None,
    ) -> str:
        running_flow_matching = sum(
            info["flow_matching_loss"] for info in training_information[-10:]
        ) / len(training_information[-10:])
        description = (
            f"Epoch: {epoch_idx}, Flow Loss: {running_flow_matching:.4f}"
        )
        if last_learning_rate is not None:
            description += f", LR: {last_learning_rate:.6f}"
        return description

    def fit_velocity_field(
        self,
        dataloader: torch.utils.data.DataLoader,
        train_parameters: TrainParameters,
        *args, **kwargs,
    ) -> Self:
        number_of_epochs_to_train = train_parameters.number_of_epochs_to_train
        verbose = train_parameters.verbose
        total_number_of_optimizer_steps = number_of_epochs_to_train * len(dataloader)

        self.warmup_y_scaler(dataloader=dataloader)
        self.Y_scaler.eval()
        self.vector_field_network.train()
        self.time_embedding_network.train()

        optimizer = torch.optim.AdamW(
            params=list(self.vector_field_network.parameters())
            + list(self.time_embedding_network.parameters()),
            **train_parameters.optimizer_parameters,
        )

        scheduler = None
        if train_parameters.scheduler_parameters:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=total_number_of_optimizer_steps,
                **train_parameters.scheduler_parameters,
            )
            
        training_information: list[dict] = []
        training_information_per_epoch: list[dict] = []

        progress_bar = trange(
            1,
            number_of_epochs_to_train + 1,
            desc="Training",
            disable=not verbose,
        )

        for epoch_idx in progress_bar:
            start_of_epoch = time.perf_counter()
            epoch_flow_losses: list[float] = []
            epoch_jacobian_penalties: list[float] = []

            for batch_index, (x_batch, y_batch) in enumerate(dataloader):
                y_scaled = self.scale_y(y_batch)
                u_batch = torch.randn_like(y_scaled)

                interpolation_times = torch.rand(
                    y_scaled.shape[0], 1, device=y_scaled.device, dtype=y_scaled.dtype
                )
                interpolated_state = (
                    interpolation_times * y_scaled + (1.0 - interpolation_times) * u_batch
                )

                optimizer.zero_grad()
                vector_field_prediction = self.predict_vector_field(
                    state=interpolated_state,
                    condition=x_batch,
                    interpolation_times=interpolation_times,
                )
                target_velocity = y_scaled - u_batch

                flow_matching_loss = (
                    (target_velocity - vector_field_prediction).pow(2).sum(dim=-1).mean()
                )
                flow_matching_loss.backward()

                torch.nn.utils.clip_grad.clip_grad_norm_(
                    self.vector_field_network.parameters(), max_norm=1
                )
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    self.time_embedding_network.parameters(), max_norm=1
                )
                
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                flow_matching_loss_value = flow_matching_loss.item()
                epoch_flow_losses.append(flow_matching_loss_value)
                training_information.append(
                    {
                        "flow_matching_loss": flow_matching_loss_value,
                        "batch_index": batch_index,
                        "epoch_index": epoch_idx,
                    }
                )

                if verbose:
                    last_learning_rate = (
                        scheduler.get_last_lr()[0] if scheduler is not None else None
                    )
                    progress_bar.set_description(
                        self.make_progress_bar_message(
                            training_information=training_information,
                            epoch_idx=epoch_idx,
                            last_learning_rate=last_learning_rate,
                        )
                    )

            training_information_per_epoch.append(
                {
                    "flow_matching_loss": torch.tensor(epoch_flow_losses).mean().item(),
                    "jacobian_penalty": torch.tensor(epoch_jacobian_penalties).mean().item(),
                    "epoch_training_time": time.perf_counter() - start_of_epoch,
                }
            )

        progress_bar.close()
        self.model_information_dict["number_of_epochs_to_train"] = number_of_epochs_to_train
        self.model_information_dict["training_batch_size"] = dataloader.batch_size
        self.model_information_dict["training_information"] = training_information_per_epoch
        return self
    
    def fit(
        self,
        dataloader: torch.utils.data.DataLoader,
        train_parameters: TrainParameters,
        *args,
        **kwargs,
    ) -> "FlowMatchingQuantile":
        self.fit_velocity_field(
            *args,
            dataloader=dataloader,
            train_parameters=train_parameters,
            **kwargs,
        )
        return self
    
    @torch.no_grad()
    def push_y_given_x(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        number_of_evaluations: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        x_tensor = x
        y_scaled = self.scale_y(y)
        number_of_evaluations = 100 if number_of_evaluations is None else number_of_evaluations

        dt = torch.full(
            (y_scaled.shape[0], 1),
            1.0 / number_of_evaluations,
            device=y_scaled.device,
            dtype=y_scaled.dtype,
        )
        state = y_scaled

        for evaluation_index in range(number_of_evaluations):
            interpolation_times = 1.0 - (evaluation_index + 0.5) * dt
            vector_field_prediction = self.predict_vector_field(
                state=state,
                condition=x_tensor,
                interpolation_times=interpolation_times,
            )
            state = state - dt * vector_field_prediction

        return state.detach()

    @torch.no_grad()
    def push_u_given_x(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
        number_of_evaluations: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        x_tensor = x
        state = u
        number_of_evaluations = 100 if number_of_evaluations is None else number_of_evaluations

        dt = torch.full(
            (state.shape[0], 1),
            1.0 / number_of_evaluations,
            device=state.device,
            dtype=state.dtype,
        )
        for evaluation_index in range(number_of_evaluations):
            interpolation_times = (evaluation_index + 0.5) * dt
            vector_field_prediction = self.predict_vector_field(
                state=state,
                condition=x_tensor,
                interpolation_times=interpolation_times,
            )
            state = state + dt * vector_field_prediction

        return self.unscale_y(state).detach()

    def save(self, path: str) -> None:
        torch.save(
            {
                "init_dict": self.init_dict,
                "state_dict": self.state_dict(),
                "model_information_dict": self.model_information_dict,
            },
            path,
        )

    def load(
        self,
        path: str,
        map_location: torch.device = torch.device("cpu"),
    ) -> Self:
        data = torch.load(path, map_location=map_location)
        self.load_state_dict(data["state_dict"])
        self.model_information_dict = data.get("model_information_dict", {})
        return self

    @classmethod
    def load_class(
        cls,
        path: str,
        map_location: torch.device = torch.device("cpu"),
    ) -> Self:
        data = torch.load(path, map_location=map_location, weights_only=False)
        operator = cls(**data["init_dict"])
        operator.load_state_dict(data["state_dict"])
        operator.model_information_dict = data.get("model_information_dict", {})
        return operator
