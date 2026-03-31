import logging
import os
import sys

import torch

from src.config import load_config
from src.datasets.cifar import get_cifar10_loaders
from src.logs.run_logger import RunLogger
from src.models.loader import load_model
from src.plotting.plots import plot_run
from src.training.trainer import Trainer


log = logging.getLogger(__name__)


def main() -> None:
    """Run all experiments defined in a YAML configuration file."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if len(sys.argv) != 2:
        log.error("Usage: python experiment.py [CONFIG_FILE]")
        sys.exit(1)

    config = load_config(sys.argv[1])
    log.info("Loaded config: model=%s, dataset=%s", config["model"], config["dataset"])
    log.info("Experiments: %s", list(config["experiments"].keys()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    torch.manual_seed(config["runtime"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["runtime"]["seed"])

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=config["runtime"]["batch_size"],
        seed=config["runtime"]["seed"],
    )
    log.info(
        "Loaded CIFAR-10: %d train, %d val, %d test",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    os.makedirs("runs", exist_ok=True)

    epochs = config["runtime"]["epochs"]
    iters_per_epoch = len(train_loader)
    total_iterations = iters_per_epoch * epochs
    log.info(
        "Runtime: %d epochs × %d iterations/epoch = %d total iterations per experiment",
        epochs,
        iters_per_epoch,
        total_iterations,
    )

    for exp_name, exp_config in config["experiments"].items():
        log.info("--- Running experiment: %s ---", exp_name)
        run_dir = os.path.join("runs", exp_name)
        os.makedirs(run_dir, exist_ok=True)

        model = load_model(config["model"])
        run_logger = RunLogger(run_dir, exp_config["optimizer"], exp_config["scheduler"])
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer_config=exp_config["optimizer"],
            scheduler_config=exp_config["scheduler"],
            epochs=config["runtime"]["epochs"],
            device=device,
            logger=run_logger,
        )
        trainer.run()
        run_logger.finalize()
        log.info("Logs written to %s/log.json", run_dir)
        plot_run(run_dir)
        log.info("Plots written to %s/", run_dir)


if __name__ == "__main__":
    main()
