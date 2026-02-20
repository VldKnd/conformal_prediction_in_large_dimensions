from __future__ import annotations

from typing import Self, Literal, Optional, Any

import torch
import torch.nn as nn
from tqdm import trange

import pydantic

from classes.protocol import PushForwardOperator
from classes.training import TrainParameters
from pushforward_operators.picnn import FFNN

# SDE - Stohastic Differential Equation

Direction = Literal["forward", "backward"]

class IterativeMarkovianFittingParameters(pydantic.BaseModel):
    number_of_markovian_projections: int = pydantic.Field(default=10)
    number_of_training_iterations: int = pydantic.Field(default=1000)
    number_of_steps_in_sde: int = pydantic.Field(default=100)
    noise_sigma_in_sde: float = pydantic.Field(default=0.5)

TIME_EPSILON = 1e-5
DEFAULT_PARAMETERS = IterativeMarkovianFittingParameters(
    number_of_markovian_projections=10,
    number_of_training_iterations=1000,
    number_of_steps_in_sde=100,
    noise_sigma_in_sde=0.5,
)

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

        self.forward_time_embedding_network = nn.Sequential(
            nn.Linear(1, feature_dimension),
            nn.Tanh(),
        )
        self.forward_drift_network = FFNN(
            feature_dimension=2 * feature_dimension,
            response_dimension=response_dimension,
            hidden_dimension=hidden_dimension,
            number_of_hidden_layers=number_of_hidden_layers,
            output_dimension=response_dimension,
            activation_function=torch.nn.Softplus(),
        )

        self.backward_time_embedding_network = nn.Sequential(
            nn.Linear(1, feature_dimension),
            nn.Tanh(),
        )
        self.backward_drift_network = FFNN(
            feature_dimension=2 * feature_dimension,
            response_dimension=response_dimension,
            hidden_dimension=hidden_dimension,
            number_of_hidden_layers=number_of_hidden_layers,
            output_dimension=response_dimension,
            activation_function=torch.nn.Softplus(),
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

    def predict_drift(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        direction: Direction,
        time: torch.Tensor,
    ) -> torch.Tensor:
        if direction == "forward":
            time_embedding = self.forward_time_embedding_network(time)
            full_condition = torch.cat([condition, time_embedding], dim=-1)
            return self.forward_drift_network(condition=full_condition, tensor=state)

        elif direction == "backward":
            time_embedding = self.backward_time_embedding_network(time)
            full_condition = torch.cat([condition, time_embedding], dim=-1)
            return self.backward_drift_network(condition=full_condition, tensor=state)
        else:
            raise ValueError(f"Unknown direction: {direction}")

    def generate_time_batch(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        random_time = torch.rand(
            (batch_size, 1),
            device=device,
            dtype=dtype
        )
        return random_time * (1.0 - 2.0 * TIME_EPSILON) + TIME_EPSILON

    @torch.no_grad()
    def get_train_tuple(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        noise_sigma_in_sde: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time = self.generate_time_batch(
            batch_size=start.shape[0],
            device=start.device,
            dtype=start.dtype
        )
        
        noise = torch.randn_like(start)

        interpolation = time * end + (1.0 - time) * start
        interpolation = interpolation + noise_sigma_in_sde * torch.sqrt(time * (1.0 - time)) * noise
        velocity = end - start - noise_sigma_in_sde * torch.sqrt(time / (1.0 - time)) * noise

        return interpolation, time, velocity
    
    @torch.no_grad()
    def sample_sde(
        self,
        start: torch.Tensor,
        condition: torch.Tensor,
        direction: Direction,
        number_of_steps_in_sde: int,
        noise_sigma_in_sde: float,
    ) -> torch.Tensor:
        self.eval()
        time_delta = 1.0 / float(number_of_steps_in_sde)
        batch_size = start.shape[0]
        device, dtype = start.device, start.dtype
        state = start.detach().clone()

        for i in range(number_of_steps_in_sde):
            timestep = float(i + 0.5) * time_delta
            time = torch.full((batch_size, 1), timestep, device=device, dtype=dtype)

            drift = self.predict_drift(
                direction=direction,
                state=state,
                condition=condition,
                time=time
            )

            state = state + drift * time_delta
            state = state + noise_sigma_in_sde * torch.randn_like(state) * (time_delta ** 0.5)

        return state.detach()

    @torch.no_grad()
    def generate_coupled_endpoints(
        self,
        x_tensor: torch.Tensor,
        y_samples: torch.Tensor,
        previous_model: Optional["SchrodingerBridgeQuantile"],
        direction: Direction,
        number_of_steps_in_sde: int,
        noise_sigma_in_sde: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        if previous_model is None:
            if direction == "forward":
                return torch.randn_like(y_samples), y_samples
            return y_samples, torch.randn_like(y_samples)

        if direction == "backward":
            u_samples = torch.randn_like(y_samples)   
            y_samples = previous_model.sample_sde(
                direction="forward",
                start=u_samples,
                condition=x_tensor,
                number_of_steps_in_sde=number_of_steps_in_sde,
                noise_sigma_in_sde=noise_sigma_in_sde
            )
            return y_samples, u_samples

        else:
            u_samples = previous_model.sample_sde(
                direction="backward",
                start=y_samples,
                condition=x_tensor,
                number_of_steps_in_sde=number_of_steps_in_sde,
                noise_sigma_in_sde=noise_sigma_in_sde
            )
            return u_samples, y_samples

    def make_progress_bar_message(
        self,
        direction: Direction,
        projection_iteration: int,
        step: int,
        losses: list[float],
        last_learning_rate: Optional[float] = None,
    ) -> str:
        window = losses[-50:] if len(losses) >= 50 else losses
        running = sum(window) / max(1, len(window))
        msg = f"IMF {projection_iteration} [{direction}] step {step}, loss {running:.4f}"
        if last_learning_rate is not None:
            msg += f", LR {last_learning_rate:.6f}"
        return msg

    def create_optimizer_and_scheduler(
            self,
            optimizer_parameters: dict[str, Any],
            scheduler_parameters: dict[str, Any],
            total_steps: int,
            direction: Direction
        ):

        if direction == "forward":    
            time_embedding_network_optimizer = torch.optim.AdamW(
                self.forward_time_embedding_network.parameters(),
                **optimizer_parameters
            )
            drift_network_optimizer = torch.optim.AdamW(
                self.forward_drift_network.parameters(),
                **optimizer_parameters,
            )
        else:
            time_embedding_network_optimizer = torch.optim.AdamW(
                self.backward_time_embedding_network.parameters(),
                **optimizer_parameters
            )
            drift_network_optimizer = torch.optim.AdamW(
                self.backward_drift_network.parameters(),
                **optimizer_parameters,
            )

        time_embedding_network_scheduler = None
        drift_network_scheduler = None

        if scheduler_parameters:
            time_embedding_network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=time_embedding_network_optimizer,
                T_max=2*total_steps,
                **scheduler_parameters,
            )
            drift_network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=drift_network_optimizer,
                T_max=total_steps,
                **scheduler_parameters,
            )
            

        return (
            time_embedding_network_optimizer,
            drift_network_optimizer,
            time_embedding_network_scheduler,
            drift_network_scheduler
        )

    def clip_gradients(self, direction: Direction):
        if direction == "forward":
            nn.utils.clip_grad_norm_(self.forward_time_embedding_network.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(self.forward_drift_network.parameters(), max_norm=1.0)

        else:
            nn.utils.clip_grad_norm_(self.backward_time_embedding_network.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(self.backward_drift_network.parameters(), max_norm=1.0)

    def fit(
        self,
        dataloader: torch.utils.data.DataLoader,
        train_parameters: TrainParameters,
        iterative_markovian_fitting_parameters: IterativeMarkovianFittingParameters = DEFAULT_PARAMETERS,
        *args,
        **kwargs,
    ) -> Self:
        self.warmup_y_scaler(dataloader)
        self.Y_scaler.eval()

        self.forward_drift_network.train()
        self.backward_drift_network.train()
        self.forward_time_embedding_network.train()
        self.backward_time_embedding_network.train()

        total_steps = iterative_markovian_fitting_parameters.number_of_training_iterations
        verbose = train_parameters.verbose

        history = []
        previous_model: Optional[SchrodingerBridgeQuantile] = None

        for projection_iteration in range(1, iterative_markovian_fitting_parameters.number_of_markovian_projections + 1):
            
            for direction in ("forward", "backward"):

                ( 
                    time_embedding_network_optimizer,
                    drift_network_optimizer,
                    time_embedding_network_scheduler,
                    drift_network_scheduler
                ) = self.create_optimizer_and_scheduler(
                    optimizer_parameters=train_parameters.optimizer_parameters,
                    scheduler_parameters=train_parameters.scheduler_parameters,
                    total_steps=total_steps,
                    direction=direction
                )

                losses: list[float] = []

                dataloader_iterator = iter(dataloader)

                progress_bar = trange(
                    1,
                    iterative_markovian_fitting_parameters.number_of_training_iterations + 1,
                    desc=f"Training IMF {projection_iteration} [{direction}]",
                    disable=not verbose,
                )

                for step in progress_bar:
                    try:
                        x_batch, y_batch = next(dataloader_iterator)
                    except StopIteration:
                        dataloader_iterator = iter(dataloader)
                        x_batch, y_batch = next(dataloader_iterator)

                    y_scaled = self.scale_y(y_batch)

                    start, end = self.generate_coupled_endpoints(
                        x_tensor=x_batch,
                        y_samples=y_scaled,
                        previous_model=previous_model,
                        direction=direction,
                        number_of_steps_in_sde=iterative_markovian_fitting_parameters.number_of_steps_in_sde,
                        noise_sigma_in_sde=iterative_markovian_fitting_parameters.noise_sigma_in_sde
                    )

                    interpolation, time, velocity = self.get_train_tuple(
                        start=start,
                        end=end,
                        noise_sigma_in_sde=iterative_markovian_fitting_parameters.noise_sigma_in_sde,
                    )

                    time_embedding_network_optimizer.zero_grad()
                    drift_network_optimizer.zero_grad()

                    vector_field_prediction = self.predict_drift(
                        direction=direction,
                        state=interpolation,
                        condition=x_batch,
                        time=time
                    )

                    markovian_fitting_loss = (velocity - vector_field_prediction).pow(2).sum(dim=-1).mean()

                    markovian_fitting_loss.backward()

                    self.clip_gradients(direction)

                    time_embedding_network_optimizer.step()
                    drift_network_optimizer.step()

                    if drift_network_scheduler is not None:
                        time_embedding_network_scheduler.step()
                        drift_network_scheduler.step()

                    loss_val = float(markovian_fitting_loss.item())
                    losses.append(loss_val)

                    if verbose:
                        last_learning_rate = (
                            drift_network_scheduler.get_last_lr()[0]
                            if drift_network_scheduler is not None else None
                        )

                        progress_bar_message = self.make_progress_bar_message(
                            direction=direction,
                            projection_iteration=projection_iteration,
                            step=step,
                            losses=losses,
                            last_learning_rate=last_learning_rate,
                        )

                        progress_bar.set_description(progress_bar_message)

                progress_bar.close()

                history.append(
                    {
                        "outer_iteration": projection_iteration,
                        "direction": direction,
                        "mean_loss": float(torch.tensor(losses).mean().item()) if losses else float("nan"),
                    }
                )
                previous_model = self.clone(device=next(self.parameters()).device)

            iterative_markovian_fitting_parameters.noise_sigma_in_sde = max(
                1e-2,
                iterative_markovian_fitting_parameters.noise_sigma_in_sde / 2
            )

        self.model_information_dict.update(
            {
                "iterative_markovian_fitting_parameters": iterative_markovian_fitting_parameters.model_dump(),
                "training_information": history,
                "training_batch_size": dataloader.batch_size,
            }
        )
        return self

    @torch.no_grad()
    def clone(self, device: torch.device) -> "SchrodingerBridgeQuantile":
        copy_model = SchrodingerBridgeQuantile(**self.init_dict).to(device=device)
        copy_model.load_state_dict(self.state_dict())
        copy_model.Y_scaler.load_state_dict(self.Y_scaler.state_dict())
        copy_model.eval()
        for parameter in copy_model.parameters():
            parameter.requires_grad_(False)
        return copy_model

    @torch.no_grad()
    def push_u_given_x(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
        number_of_evaluations: int = 100,
    ) -> torch.Tensor:
        self.eval()

        y_scaled = self.sample_sde(
            start=u,
            condition=x,
            direction="forward",
            number_of_steps_in_sde=number_of_evaluations,
            noise_sigma_in_sde=0.
        )
        
        return self.unscale_y(y_scaled).detach()
    
    @torch.no_grad()
    def push_y_given_x(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        number_of_evaluations: int = 100
    ) -> torch.Tensor:
        self.eval()
        y_scaled = self.scale_y(y)

        return self.sample_sde(
            start=y_scaled,
            condition=x,
            direction="backward",
            number_of_steps_in_sde=number_of_evaluations,
            noise_sigma_in_sde=0.
        )

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