from typing import Any, List
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig, MISSING
import hydra


@dataclass
class Experiment:
    model: str = MISSING
    nrof_epochs: int = 30
    lr: float = 0.001

@dataclass
class Resnet18Experiment(Experiment):
    model: str = "resnet18"
    batch_size: int = 128


@dataclass
class Resnet50Experiment(Experiment):
    model: str = "resnet50"
    lr_scheduler: str = "cosine"


DEFAULT = [
    {"experiment": "resnet18"},
    "_self_",
]


@dataclass
class MyConfig:
    experiment: Any

@dataclass
class ListConfig:
    defaults: List[Any] = field(default_factory=lambda: DEFAULT)


cs = ConfigStore.instance()
cs.store(name="config", node=MyConfig)
cs.store(name="list_config", node=ListConfig)
cs.store(group="experiment", name="resnet18", node=Resnet18Experiment)
cs.store(group="experiment", name="resnet50", node=Resnet50Experiment)


# @hydra.main(config_path=None, config_name="config", version_base=None)
@hydra.main(config_path=None, config_name="list_config", version_base=None)
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