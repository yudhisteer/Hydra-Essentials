from typing import Any, List, Optional
from pydantic.dataclasses import dataclass
from pydantic import field_validator

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig, MISSING, omegaconf
import hydra


@dataclass
class ExperimentSchema:
    model: str = MISSING
    nrof_epochs: int = 30
    lr: float = 0.001
    batch_size: int = 128

    @field_validator("batch_size")
    def validate_batch_size(cls, batch_size: int) -> int:
        if batch_size % 8 != 0:
            raise ValueError("Batch size must be divisible by 8")
        return batch_size


@dataclass
class Resnet18ExperimentSchema(ExperimentSchema):
    model: str = "resnet18"


@dataclass
class Resnet50ExperimentSchema(ExperimentSchema):
    model: str = "resnet50"
    lr_scheduler: Optional[str] = None


@dataclass
class ConfigSchema:
    experiment: ExperimentSchema



cs = ConfigStore.instance()
cs.store(name="config_schema", node=ConfigSchema)
cs.store(group="experiment", name="resnet18_schema", node=Resnet18ExperimentSchema)
cs.store(group="experiment", name="resnet50_schema", node=Resnet50ExperimentSchema)



@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    OmegaConf.to_object(config)
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