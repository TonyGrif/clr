import json
import os
from typing import List, Dict

import pandas as pd
import seaborn.objects as so


def plot_run(run_dir: str) -> None:
    """Generate training plots from a run's log.json.

    Produces three PNG plots saved to run_dir:
    - accuracy_train.png: iteration vs training accuracy
    - accuracy_val.png: epoch vs validation accuracy
    - learning_rate.png: iteration vs learning rate

    Args:
        run_dir: Directory containing log.json; plots are saved here.
    """
    path = os.path.join(run_dir, "log.json")
    with open(path, "r") as f:
        log = json.load(f)

    iterations = log["iterations"]
    val_iterations = [i for i in iterations if i["val_accuracy"] is not None]

    _plot_train_accuracy(iterations, run_dir)
    _plot_val_accuracy(val_iterations, run_dir)
    _plot_learning_rate(iterations, run_dir)


def _plot_train_accuracy(iterations: List[Dict], run_dir: str) -> None:
    """Plot training accuracy over iterations.

    Args:
        iterations: List of per-iteration log entries.
        run_dir: Directory to save the plot.
    """
    df = pd.DataFrame(iterations)[["iteration", "train_accuracy"]]
    (
        so.Plot(df, x="iteration", y="train_accuracy")
        .add(so.Line())
        .label(x="Iteration", y="Training Accuracy", title="Training Accuracy vs Iteration")
        .save(os.path.join(run_dir, "accuracy_train.png"))
    )


def _plot_val_accuracy(val_iterations: List[Dict], run_dir: str) -> None:
    """Plot validation accuracy at epoch boundaries.

    Args:
        val_iterations: Iteration entries where val_accuracy is not None.
        run_dir: Directory to save the plot.
    """
    df = pd.DataFrame(val_iterations)[["iteration", "val_accuracy"]]
    (
        so.Plot(df, x="iteration", y="val_accuracy")
        .add(so.Line())
        .add(so.Dot())
        .label(x="Iteration", y="Validation Accuracy", title="Validation Accuracy vs Iteration")
        .save(os.path.join(run_dir, "accuracy_val.png"))
    )


def _plot_learning_rate(iterations: List[Dict], run_dir: str) -> None:
    """Plot learning rate over iterations.

    Args:
        iterations: List of per-iteration log entries.
        run_dir: Directory to save the plot.
    """
    df = pd.DataFrame(iterations)[["iteration", "lr"]]
    (
        so.Plot(df, x="iteration", y="lr")
        .add(so.Line())
        .label(x="Iteration", y="Learning Rate", title="Learning Rate vs Iteration")
        .save(os.path.join(run_dir, "learning_rate.png"))
    )
