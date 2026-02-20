from __future__ import annotations

import logging
from pushforward_operators.utilities import rolling_mean
from datetime import datetime
from pathlib import Path
from uuid import uuid4

class TrainingLogger:
    def __init__(
        self,
        *,
        model_name: str,
    ) -> None:
        self.batch_log_interval = 50
        self.metrics: dict[str, list[float]] = {}
        self.model_name = model_name
        self._handlers: list[logging.Handler] = []

        logger_name = (
            f"{model_name}"
            f".{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.{uuid4().hex[:8]}"
        )
        self.logger = logging.getLogger(logger_name)
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        
        log_dir = Path("attic") / "log" / model_name
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = log_dir / f"{timestamp}_{uuid4().hex[:8]}.log"
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self._handlers.append(file_handler)

    def update_metrics(self, metric_values: dict[str, float]):
        for key, value in metric_values.items():
            if key in self.metrics:
                self.metrics[key].append(value)
            else:
                self.metrics[key] = [value]

    def make_message(self, window: int = 10):
        return ", ".join(
            f"{key} : {rolling_mean(values=values, window=window):.3f}"
            for key, values in self.metrics.items()
        )

    def log_batch(
        self,
        *,
        epoch_index: int,
        batch_index: int,
        metric_values: dict[str, float],
        learning_rate: float | None = None,
    ) -> str:
        
        self.update_metrics(metric_values=metric_values)
        
        metric_message = self.make_message()

        if learning_rate is not None:
            metric_message += f", lr={learning_rate:.6f}"

        if batch_index % self.batch_log_interval != 0:
            return metric_message

        self.logger.info(
            "BATCH epoch=%d batch=%d %s",
            epoch_index,
            batch_index,
            metric_message,
        )

        return metric_message

    def log_epoch(self, *, epoch_index: int) -> None:
        metric_message = self.make_message(window=-1)
        self.logger.info(
            "EPOCH epoch=%d %s",
            epoch_index,
            metric_message,
        )

    def close(self) -> None:
        for handler in self._handlers:
            self.logger.removeHandler(handler)
            handler.flush()
            handler.close()
        self._handlers.clear()

