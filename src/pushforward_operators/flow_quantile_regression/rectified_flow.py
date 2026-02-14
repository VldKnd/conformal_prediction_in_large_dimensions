from __future__ import annotations


import copy
import torch
import time
from tqdm import trange
from typing import Self
from classes.training import TrainParameters
from .flow_matching import FlowMatchingQuantile

class RectifiedFlowQuantile(FlowMatchingQuantile):
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

    def rectify_velocity_field(
        self,
        dataloader: torch.utils.data.DataLoader,
        train_parameters: TrainParameters,
        *args, **kwargs
    ) -> Self:

        previous_flow_model = copy.deepcopy(self)
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
                flow_matching_loss.backward()
                
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
        number_of_rectifying_operations: int = 1,
        **kwargs,
    ) -> Self:
        
        self.fit_velocity_field(
            dataloader=dataloader,
            train_parameters=train_parameters,
            **kwargs,
        )

        for _ in range(number_of_rectifying_operations):
            train_parameters.number_of_epochs_to_train = train_parameters.number_of_epochs_to_train // 2
            self.rectify_velocity_field(
                dataloader=dataloader,
                train_parameters=train_parameters,
                **kwargs,
            )

        return self