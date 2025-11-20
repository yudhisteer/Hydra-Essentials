from typing import Optional
from dataclasses import dataclass, field

from rich import print
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.config_store import ConfigStore


@dataclass
class TrainingConfig:
    model: str = "resnet18"
    epochs: int = 100
    lr: float = 0.001

    # Optional fields - can be None or a value
    pretrained_weights: Optional[str] = None
    lr_scheduler: Optional[str] = None
    early_stopping_patience: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None


@dataclass
class Config:
    training: TrainingConfig = field(default_factory=TrainingConfig)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_path=None, config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

    # Demonstrate None checking
    print("\n--- Checking Optional Fields ---")
    if config.training.pretrained_weights is not None:
        print(f"Loading pretrained weights from: {config.training.pretrained_weights}")
    else:
        print("No pretrained weights specified, training from scratch")

    if config.training.lr_scheduler is not None:
        print(f"Using learning rate scheduler: {config.training.lr_scheduler}")
    else:
        print("No learning rate scheduler")

    if config.training.early_stopping_patience is not None:
        print(f"Early stopping enabled with patience: {config.training.early_stopping_patience}")
    else:
        print("Early stopping disabled")


if __name__ == "__main__":
    main()


# Example runs:
# python structured_09/optional_fields.py
# python structured_09/optional_fields.py training.pretrained_weights=/path/to/weights.pth
# python structured_09/optional_fields.py training.lr_scheduler=cosine training.early_stopping_patience=10
