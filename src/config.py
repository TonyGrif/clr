from typing import Dict

import yaml


REQUIRED_TOP_LEVEL_KEYS = {"model", "dataset", "runtime", "experiments"}
REQUIRED_RUNTIME_KEYS = {"epochs", "batch_size", "seed"}


def load_config(path: str) -> Dict:
    """Load and validate an experiment configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required keys are missing.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    missing = REQUIRED_TOP_LEVEL_KEYS - config.keys()
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    missing_runtime = REQUIRED_RUNTIME_KEYS - config["runtime"].keys()
    if missing_runtime:
        raise ValueError(f"Config 'runtime' missing required keys: {missing_runtime}")

    if not config["experiments"]:
        raise ValueError("Config must define at least one experiment.")

    return config
