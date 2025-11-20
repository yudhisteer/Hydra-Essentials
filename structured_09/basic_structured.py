from dataclasses import dataclass

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

# register the config class to the config store
cs = ConfigStore.instance()
cs.store(name="experiment_config", node=ExperimentConfig)


@hydra.main(config_path=None, config_name="experiment_config", version_base=None)
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

    # access the config values
    print("Model: ", config.model)
    print("Number of epochs: ", config.nrof_epochs)
    print("Learning rate: ", config.lr)

if __name__ == "__main__":
    main()