from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING
from rich import print
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


@dataclass
class ModelConfig:
    name: str = "resnet18"
    layers: int = 18
    pretrained: bool = False
    dropout: float = 0.5


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 100
    lr: float = 0.001
    optimizer: str = "adam"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42


def main() -> None:
    print("="*60)
    print("1. Creating config from dataclass")
    print("="*60)
    config = OmegaConf.structured(Config)
    print(OmegaConf.to_yaml(config))

    print("="*60)
    print("2. Type safety - correct type")
    print("="*60)
    config.training.batch_size = 128
    print(f"Batch size updated to: {config.training.batch_size}")

    print("\n" + "="*60)
    print("3. Type safety - wrong type (will raise error)")
    print("="*60)
    try:
        config.training.batch_size = "invalid"  # Should be int
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

    print("\n" + "="*60)
    print("4. Struct mode enabled by default")
    print("="*60)
    print(f"Is struct? {OmegaConf.is_struct(config)}")
    try:
        config.new_field = "value"
        #  structured configs still behave as if struct mode is enabled, so adding a new field raises an error.
    except Exception as e:
        print(f"Error when adding new field: {type(e).__name__}: {e}")

    print("\n" + "="*60)
    print("5. Merging with unstructured config")
    print("="*60)
    yaml_config = OmegaConf.create({
        "model": {"pretrained": True, "layers": 50},
        "training": {"epochs": 200}
    })

    merged = OmegaConf.merge(config, yaml_config)
    print("Merged config:")
    print(OmegaConf.to_yaml(merged))

    print("\n" + "="*60)
    print("6. Accessing as object")
    print("="*60)
    # Convert back to Python object
    config_obj = OmegaConf.to_object(merged)
    print(f"Type: {type(config_obj)}")
    print(f"Model name: {config_obj.model.name}")
    print(f"Training epochs: {config_obj.training.epochs}")


if __name__ == "__main__":
    main()
