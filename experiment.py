import sys

from src.config import load_config
from src.datasets.cifar import get_cifar10_loaders
from src.models.loader import load_model


def main() -> None:
    """Run all experiments defined in a YAML configuration file."""
    if len(sys.argv) != 2:
        print("Usage: python experiment.py [CONFIG_FILE]")
        sys.exit(1)

    config = load_config(sys.argv[1])
    print(f"Loaded config: model={config['model']}, dataset={config['dataset']}")
    print(f"Runtime: {config['runtime']}")
    print(f"Experiments: {list(config['experiments'].keys())}")

    model = load_model(config["model"])
    print(f"Loaded model: {config['model']}")

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=config["runtime"]["batch_size"],
        seed=config["runtime"]["seed"],
    )
    print(f"Loaded CIFAR-10: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test")


if __name__ == "__main__":
    main()
