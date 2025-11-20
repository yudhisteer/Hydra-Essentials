from typing import List, Dict, Any
from dataclasses import dataclass, field

from rich import print
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.config_store import ConfigStore


@dataclass
class AugmentationConfig:
    # List with default values
    transforms: List[str] = field(default_factory=lambda: ["resize", "normalize"])
    # Empty dict as default
    transform_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    batch_size: int = 32
    num_workers: int = 4
    # Empty list as default
    train_datasets: List[str] = field(default_factory=list)
    # Empty dict as default
    dataset_weights: Dict[str, float] = field(default_factory=dict)
    # Nested dataclass with lists/dicts
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_path=None, config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

    # Demonstrate working with lists and dicts
    print("\n--- Working with Lists and Dicts ---")
    print(f"Transforms to apply: {list(config.data.augmentation.transforms)}")
    print(f"Number of training datasets: {len(config.data.train_datasets)}")

    if config.data.dataset_weights:
        print(f"Dataset weights: {dict(config.data.dataset_weights)}")
    else:
        print("No dataset weights specified (equal weighting)")


if __name__ == "__main__":
    main()


# Example runs:
# python structured_09/lists_dicts.py
# python structured_09/lists_dicts.py data.train_datasets='[imagenet,coco]'
# python structured_09/lists_dicts.py data.dataset_weights='{imagenet:0.7,coco:0.3}'
# python structured_09/lists_dicts.py data.augmentation.transforms='[resize,crop,normalize,flip]'
