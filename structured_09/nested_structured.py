from dataclasses import dataclass, field
from rich import print
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.config_store import ConfigStore


# define the config class
@dataclass
class ExperimentConfig:
    model: str = "resnet18"
    nrof_epochs: int = 30
    lr: float = 0.001


@dataclass
class LossConfig:
    name: str = "arcface"
    margin: float = 0.8


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    loss: LossConfig = field(default_factory=LossConfig)


# register the config class to the config store
cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_path=None, config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    main()