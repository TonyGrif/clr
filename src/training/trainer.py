import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.logs.run_logger import RunLogger

logger = logging.getLogger(__name__)


class Trainer:
    """Runs a single experiment: trains a model and logs results."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer_config: Dict,
        scheduler_config: Dict,
        epochs: int,
        device: torch.device,
        logger: RunLogger,
    ) -> None:
        """Set up trainer with model, data, optimizer, scheduler, and logger.

        Args:
            model: The neural network to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            optimizer_config: Dict with 'name' and optional kwargs.
            scheduler_config: Dict with 'name' and optional kwargs.
            epochs: Number of training epochs.
            device: Device to run training on.
            logger: RunLogger instance for this experiment.
        """
        self._model = model.to(device)
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._epochs = epochs
        self._device = device
        self._logger = logger
        self._criterion = nn.CrossEntropyLoss()

        opt_cls = getattr(torch.optim, optimizer_config["name"])
        opt_kwargs = {k: v for k, v in optimizer_config.items() if k != "name"}
        self._optimizer = opt_cls(self._model.parameters(), **opt_kwargs)

        self._scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        if scheduler_config["name"] != "constant":
            sched_cls = getattr(torch.optim.lr_scheduler, scheduler_config["name"])
            sched_kwargs = {k: v for k, v in scheduler_config.items() if k != "name"}
            self._scheduler = sched_cls(self._optimizer, **sched_kwargs)

    def run(self) -> None:
        """Execute the full training loop across all epochs."""
        iteration = 1

        for epoch in range(1, self._epochs + 1):
            self._model.train()
            running_correct = 0
            running_total = 0

            for inputs, labels in self._train_loader:
                inputs, labels = inputs.to(self._device), labels.to(self._device)

                self._optimizer.zero_grad()
                outputs = self._model(inputs)
                loss = self._criterion(outputs, labels)
                loss.backward()
                self._optimizer.step()

                if self._scheduler is not None:
                    self._scheduler.step()

                preds = outputs.argmax(dim=1)
                running_correct += (preds == labels).sum().item()
                running_total += labels.size(0)

                lr = self._optimizer.param_groups[0]["lr"]
                train_acc = running_correct / running_total

                self._logger.log_iteration(
                    iteration=iteration,
                    lr=lr,
                    loss=loss.item(),
                    train_accuracy=train_acc,
                    val_accuracy=None,
                )
                iteration += 1

            val_acc = self._validate()
            # Backfill val_accuracy for the last iteration of this epoch
            self._logger._iterations[-1]["val_accuracy"] = val_acc

            logger.info("Epoch %d/%d — val_acc: %.4f", epoch, self._epochs, val_acc)

    def _validate(self) -> float:
        """Run a full validation pass and return accuracy.

        Returns:
            Fraction of correctly classified validation samples.
        """
        self._model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self._val_loader:
                inputs, labels = inputs.to(self._device), labels.to(self._device)
                outputs = self._model(inputs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total
