import json
import os
from typing import Dict, List, Optional


class RunLogger:
    """Manages JSON logging for a single experiment run."""

    def __init__(
        self,
        run_dir: str,
        optimizer_config: Dict,
        scheduler_config: Dict,
    ) -> None:
        """Initialize the logger and store run header information.

        Args:
            run_dir: Directory where log.json will be written.
            optimizer_config: Optimizer config dict from YAML.
            scheduler_config: Scheduler config dict from YAML.
        """
        self._run_dir = run_dir
        self._header = {
            "optimizer": optimizer_config,
            "scheduler": scheduler_config,
        }
        self._iterations: List[Dict] = []

    def log_iteration(
        self,
        iteration: int,
        lr: float,
        loss: float,
        train_accuracy: float,
        val_accuracy: Optional[float],
    ) -> None:
        """Append one iteration record to the log.

        Args:
            iteration: Global iteration counter (1-indexed).
            lr: Current learning rate.
            loss: Batch loss value.
            train_accuracy: Running training accuracy for the current epoch.
            val_accuracy: Validation accuracy, or None if mid-epoch.
        """
        self._iterations.append({
            "iteration": iteration,
            "lr": lr,
            "loss": loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        })

    def finalize(self) -> None:
        """Write the complete log to disk as log.json."""
        log = {**self._header, "iterations": self._iterations}
        path = os.path.join(self._run_dir, "log.json")
        with open(path, "w") as f:
            json.dump(log, f, indent=2)
