from __future__ import annotations


import copy
import torch
import time
from functools import partial
from tqdm import trange
from typing import Self
from classes.training import TrainParameters
from .flow_matching import FlowMatchingQuantile

class RectifiedJacobianFlowQuantile(FlowMatchingQuantile):
    def __init__(
        self,
        feature_dimension: int,
        response_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
    ):
        super().__init__(
            feature_dimension=feature_dimension,
            response_dimension=response_dimension,
            hidden_dimension=hidden_dimension,
            number_of_hidden_layers=number_of_hidden_layers
        )
        self.model_information_dict = {
            "class_name": "RectifiedFlowQuantile",
        }

    def get_jacobian_vector_product(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        interpolation_times: torch.Tensor,
        random_directions: torch.Tensor,
    ) -> torch.Tensor:

        def get_conditional_vector_field_as_a_function(_condition: torch.Tensor):
            return partial(self.vector_field_network, _condition)

        def calculate_jacobian_vector_product_per_tensor(
            _condition: torch.Tensor,
            _state: torch.Tensor,
            _random_direction: torch.Tensor
        ):
            assert (_condition.ndim <= 2 and _state.ndim <= 2 and _random_direction.ndim <= 2), \
                (
                    f"{_condition.ndim=} ( should be less then 2 ) "
                    f"{_state.ndim=} ( should be less then 2 ) "
                    f"{_random_direction.ndim=} ( should be less then 2 )"
                )
        
            _condition = _condition if _condition.ndim != 1 else _condition.unsqueeze(0)
            _state = _state if _state.ndim != 1 else _state.unsqueeze(0)
            _random_direction = _random_direction if _random_direction.ndim != 1 else _random_direction.unsqueeze(0)

            conditional_vector_field_as_a_function = get_conditional_vector_field_as_a_function(_condition=_condition)
            _, jacobian_vector_product = torch.func.jvp(
                conditional_vector_field_as_a_function,
                (_state, ),
                (_random_direction, )
            )
            return jacobian_vector_product.squeeze(0)
        
        time_embedding = self.time_embedding_network(interpolation_times)
        full_condition = torch.cat([condition, time_embedding], dim=1)

        return torch.vmap(
            lambda _condition, _state, _random_directions:
                torch.vmap(
                    lambda _random_direction: \
                        calculate_jacobian_vector_product_per_tensor(
                            _condition = _condition,
                            _state = _state, 
                            _random_direction = _random_direction
                        ),
                    in_dims=-1, out_dims=-1
                )(_random_directions)
        )(full_condition, state, random_directions)


    def get_vector_jacobian_product(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        interpolation_times: torch.Tensor,
        random_directions: torch.Tensor,
    ) -> torch.Tensor:
        def get_conditional_vector_field_as_a_function(_condition: torch.Tensor):
            return partial(self.vector_field_network, _condition)

        def calculate_vector_jacobian_product_per_direction_batch(
            _condition: torch.Tensor,
            _state: torch.Tensor,
            _random_directions: torch.Tensor
        ):
            assert (_condition.ndim <= 2 and _state.ndim <= 2 and _random_directions.ndim == _state.ndim + 1), \
                (
                    f"{_condition.ndim=} ( should be less then 2 ) "
                    f"{_state.ndim=} ( should be less then 2 ) "
                    f"{_random_directions.ndim=} ( should be one more then _state and _condition dim )"
                )
        
            _condition = _condition if _condition.ndim != 1 else _condition.unsqueeze(0)
            _state = _state if _state.ndim != 1 else _state.unsqueeze(0)

            conditional_vector_field_as_a_function = get_conditional_vector_field_as_a_function(_condition=_condition)
            _, vector_jacobian_product_function = torch.func.vjp(
                conditional_vector_field_as_a_function,
                _state
            )
            vector_jacobian_products = torch.vmap(
                lambda _random_direction: \
                    vector_jacobian_product_function(
                        _random_direction.unsqueeze(0)
                    )[0].squeeze(0),
                in_dims=-1, out_dims=-1
            )(_random_directions)
            return vector_jacobian_products
        
        time_embedding = self.time_embedding_network(interpolation_times)
        full_condition = torch.cat([condition, time_embedding], dim=1)

        return torch.vmap(
            lambda _condition, _state, _random_directions:
                calculate_vector_jacobian_product_per_direction_batch(
                    _condition=_condition,
                    _state=_state,
                    _random_directions=_random_directions
                )
        )(full_condition, state, random_directions)


    def calculate_jacobian_symmetry_penalty(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        interpolation_times: torch.Tensor,
        hutchinson_estimation_batch_size: int = 64,
    ) -> torch.Tensor:
        random_directions = torch.randn(*state.shape, hutchinson_estimation_batch_size)

        jacobian_vector_product = self.get_jacobian_vector_product(
            state=state,
            condition=condition,
            interpolation_times=interpolation_times,
            random_directions=random_directions
        )

        vector_jacobian_product = self.get_vector_jacobian_product(
            state=state,
            condition=condition,
            interpolation_times=interpolation_times,
            random_directions=random_directions
        )

        return (jacobian_vector_product - vector_jacobian_product).pow(2).mean()


    def make_progress_bar_message(
        self,
        training_information: list[dict],
        epoch_idx: int,
        last_learning_rate: float | None = None,
    ) -> str:
        recent_info = training_information[-10:] if len(training_information) >= 10 else training_information
        running_flow_matching = (
            sum(info["flow_matching_loss"] for info in recent_info) / len(recent_info)
        )

        if any("jacobian_penalty" in info for info in recent_info):
            running_jacobian_penalty = (
                sum(info["jacobian_penalty"] for info in recent_info) / len(recent_info)
            )
        else:
            running_jacobian_penalty = None

        description = f"Epoch: {epoch_idx}, Flow Loss: {running_flow_matching:.4f}"
        if running_jacobian_penalty is not None:
            description += f", Jacobian Penalty: {running_jacobian_penalty:.4f}"
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
        if train_parameters.scheduler_parameters:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=total_number_of_optimizer_steps,
                **train_parameters.scheduler_parameters,
            )
        else:
            scheduler = None

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

                jacobian_symmetry_penalty = self.calculate_jacobian_symmetry_penalty(
                    state=interpolated_state,
                    condition=x_batch,
                    interpolation_times=interpolation_times
                )

                total_loss = flow_matching_loss + 0.1 * jacobian_symmetry_penalty

                total_loss.backward()

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
                jacobian_penalty_value = jacobian_symmetry_penalty.item()

                epoch_flow_losses.append(flow_matching_loss_value)
                epoch_jacobian_penalties.append(jacobian_penalty_value)

                training_information.append(
                    {
                        "flow_matching_loss": flow_matching_loss_value,
                        "jacobian_penalty": jacobian_penalty_value,
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


    def rectify_velocity_field(
        self,
        dataloader: torch.utils.data.DataLoader,
        train_parameters: TrainParameters,
        *args, **kwargs
    ) -> Self:
        previous_flow_model = self.clone()
        previous_flow_model.eval()
        
        number_of_epochs_to_train = train_parameters.number_of_epochs_to_train
        verbose = train_parameters.verbose
        total_number_of_optimizer_steps = number_of_epochs_to_train * len(dataloader)

        self.vector_field_network.train()
        self.time_embedding_network.train()

        optimizer = torch.optim.AdamW(
            params=list(self.vector_field_network.parameters())
            + list(self.time_embedding_network.parameters()),
            **train_parameters.optimizer_parameters,
        )
        if train_parameters.scheduler_parameters:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=total_number_of_optimizer_steps,
                **train_parameters.scheduler_parameters,
            )
        else:
            scheduler = None

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

            for batch_index, (x_batch, y_dataloader_batch) in enumerate(dataloader):
                u_batch = torch.randn_like(y_dataloader_batch)
                y_batch = previous_flow_model.push_u_given_x(u=u_batch, x=x_batch)
                y_scaled = self.scale_y(y_batch)
                
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

                jacobian_symmetry_penalty = self.calculate_jacobian_symmetry_penalty(
                    state=interpolated_state,
                    condition=x_batch,
                    interpolation_times=interpolation_times
                )

                total_loss = flow_matching_loss + 0.1 * jacobian_symmetry_penalty

                total_loss.backward()

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
                jacobian_penalty_value = jacobian_symmetry_penalty.item()

                epoch_flow_losses.append(flow_matching_loss_value)
                epoch_jacobian_penalties.append(jacobian_penalty_value)

                training_information.append(
                    {
                        "flow_matching_loss": flow_matching_loss_value,
                        "jacobian_penalty": jacobian_penalty_value,
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
        self.model_information_dict["number_of_epochs_to_rectify"] = number_of_epochs_to_train
        self.model_information_dict["rectifying_batch_size"] = dataloader.batch_size
        self.model_information_dict["rectifying_information"] = training_information_per_epoch
        return self
    
    def fit(
        self,
        dataloader: torch.utils.data.DataLoader,
        train_parameters: TrainParameters,
        **kwargs,
    ) -> Self:
        
        self.fit_velocity_field(
            dataloader=dataloader,
            train_parameters=train_parameters,
            **kwargs,
        )

        train_parameters.number_of_epochs_to_train = train_parameters.number_of_epochs_to_train // 2

        self.rectify_velocity_field(
            dataloader=dataloader,
            train_parameters=train_parameters,
            **kwargs,
        )

        return self
    
    def clone(self, *, detach=True, device=None, dtype=None, train_mode=None) -> Self:
        new_self = type(self)(**self.init_dict)

        if device is not None or dtype is not None:
            new_self = new_self.to(device=device, dtype=dtype)

        state_dict = self.state_dict()

        if detach:
            state_dict = {
                key: value.detach().clone()
                for key, value in state_dict.items()
            }

        new_self.load_state_dict(state_dict, strict=True)

        if train_mode is None:
            new_self.train(self.training)
        else:
            new_self.train(train_mode)

        return new_self