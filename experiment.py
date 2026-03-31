import sys

from src.config import load_config


def main() -> None:
    """Run all experiments defined in a YAML configuration file."""
    if len(sys.argv) != 2:
        print("Usage: python experiment.py [CONFIG_FILE]")
        sys.exit(1)

    config = load_config(sys.argv[1])
    print(f"Loaded config: model={config['model']}, dataset={config['dataset']}")
    print(f"Runtime: {config['runtime']}")
    print(f"Experiments: {list(config['experiments'].keys())}")


if __name__ == "__main__":
    main()
