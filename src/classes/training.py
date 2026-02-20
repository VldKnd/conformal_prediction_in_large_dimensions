from typing import Any

import pydantic


class TrainParameters(pydantic.BaseModel):
    number_of_epochs_to_train: int = pydantic.Field(default=500, ge=1)
    optimizer_parameters: dict[str, Any] = pydantic.Field(default_factory=dict)
    scheduler_parameters: dict[str, Any] = pydantic.Field(default_factory=dict)
    warmup_iterations: int = pydantic.Field(default=5, ge=0)
    verbose: bool = pydantic.Field(default=False)
