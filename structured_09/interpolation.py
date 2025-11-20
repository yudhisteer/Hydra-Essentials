from dataclasses import dataclass, field

from rich import print
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.config_store import ConfigStore


@dataclass
class ExperimentConfig:
    model: str = "resnet18"
    dataset: str = "imagenet"
    epochs: int = 100
    lr: float = 0.001


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # These will be overridden by YAML with interpolations
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_file: str = "train.log"


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


@hydra.main(config_path="configs", config_name="interpolation_config", version_base=None)
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

    # Demonstrate resolved interpolations
    print("\n--- Resolved Paths ---")
    print(f"Output directory: {config.output_dir}")
    print(f"Checkpoint directory: {config.checkpoint_dir}")
    print(f"Log file: {config.log_file}")


if __name__ == "__main__":
    main()


# Example runs:
# python structured_09/interpolation.py
# python structured_09/interpolation.py experiment.model=resnet50 experiment.dataset=coco
