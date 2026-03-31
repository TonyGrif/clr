# Changelog

## [Unreleased]

### Added
- Initial project scaffold: pyproject.toml, requirements.txt, .gitignore, CHANGELOG.md
- Configuration loading: src/config.py with YAML validation, configs/example.yaml
- Model loading: src/models/loader.py for resnet18 and densenet121 with 10-class head
- Dataset loading: src/datasets/cifar.py with 45k train / 5k val / 10k test split
- Run logger: src/logs/run_logger.py accumulates per-iteration metrics and writes log.json
- Training engine: src/training/trainer.py with dynamic optimizer/scheduler loading and per-iteration logging
- Plotting: src/plotting/plots.py generates accuracy and learning rate PNGs using seaborn objects
- Paper replication config: configs/clr_paper.yaml with baseline, triangular, and triangular2 experiments
- Model architecture: adapted resnet18 and densenet121 for CIFAR-10 with 3x3 stride-1 conv and identity max pool
- Dataset: added random horizontal flip and random crop (padding=4) augmentation to training set only
