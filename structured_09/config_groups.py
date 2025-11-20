from typing import Any
from dataclasses import dataclass


from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig
import hydra

@dataclass
class Resnet18Experiment:
    model: str = "resnet18"
    nrof_epochs: int = 30
    lr: float = 0.001


@dataclass
class Resnet50Experiment:
    model: str = "resnet50"
    nrof_epochs: int = 30
    lr: float = 0.001


@dataclass
class MyConfig:
    experiment: Any

cs = ConfigStore.instance()
cs.store(name="config", node=MyConfig)
cs.store(group="experiment", name="resnet18", node=Resnet18Experiment)
cs.store(group="experiment", name="resnet50", node=Resnet50Experiment)


@hydra.main(config_path=None, config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))



if __name__ == "__main__":
    main()


# if we run python '/home/cyudhist/workspace/Hydra-Essentials/structured_09/config_groups.py'
# we get: experiment: ???

# if we run python '/home/cyudhist/workspace/Hydra-Essentials/structured_09/config_groups.py' +experiment=resnet18
# we get

# ```
# experiment:
#   model: resnet18
#   nrof_epochs: 30
#   lr: 0.001
# ```