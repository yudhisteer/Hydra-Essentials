## 9. Structured Configs

Structured Configs allow you to define your Hydra configurations using Python dataclasses instead of (or in addition to) YAML files. This provides type safety, IDE autocompletion, and runtime validation.

### 9.1 Why Use Structured Configs?

**Benefits over YAML-only configs:**

1. **Type Safety**: Catch type errors before runtime
2. **IDE Support**: Autocompletion, refactoring, go-to-definition
3. **Validation**: Built-in type checking and custom validators
4. **Documentation**: Docstrings and type hints serve as documentation
5. **Reusability**: Python inheritance and composition patterns

**When to use Structured Configs:**
- Complex configurations with many fields
- When type safety is critical
- When you need custom validation logic
- When you want IDE support for config authoring

### 9.2 Basic Structured Config

The simplest structured config uses a single dataclass registered with the ConfigStore.

```python
#structured_09/basic_structured.py
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
```

**Key Points:**
- `config_path=None` tells Hydra there's no YAML config directory
- `config_name="experiment_config"` matches the name in `cs.store()`
- The ConfigStore is a singleton - use `ConfigStore.instance()`

Running this:

```bash
python structured_09/basic_structured.py
```

Output:

```bash
model: resnet18
nrof_epochs: 30
lr: 0.001

Model:  resnet18
Number of epochs:  30
Learning rate:  0.001
```

**Overriding from CLI:**

```bash
python structured_09/basic_structured.py model=resnet50 lr=0.0001
```

Output:

```bash
model: resnet50
nrof_epochs: 30
lr: 0.0001
```

### 9.3 Hierarchical Structured Configs

For more complex configurations, you can nest dataclasses to create hierarchical structures.

```python
#structured_09/nested_structured.py
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
```

**Important:** Use `field(default_factory=ClassName)` for nested dataclasses, not direct instantiation.

Running this:

```bash
python structured_09/nested_structured.py
```

Output:

```bash
experiment:
  model: resnet18
  nrof_epochs: 30
  lr: 0.001
loss:
  name: arcface
  margin: 0.8
```

**Overriding nested values:**

```bash
python structured_09/nested_structured.py experiment.lr=0.01 loss.margin=0.5
```

### 9.4 Config Groups with Structured Configs

You can create config groups entirely in Python by storing multiple configs under a group name.

```python
#structured_09/config_groups.py
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
```

**Key Points:**
- `group="experiment"` creates a config group
- `name="resnet18"` is the option name within that group
- `experiment: Any` allows any config to be placed there

Running without specifying experiment:

```bash
python structured_09/config_groups.py
```

Output:

```bash
experiment: ???
```

The `???` indicates a missing required value.

**Selecting a config group option:**

```bash
python structured_09/config_groups.py +experiment=resnet18
```

Output:

```bash
experiment:
  model: resnet18
  nrof_epochs: 30
  lr: 0.001
```

Note: Use `+experiment` (with `+`) because we're adding a new default, not overriding an existing one.

### 9.5 Inheritance in Structured Configs

Python inheritance allows you to create base configs with common fields and specialized configs that extend them.

```python
#structured_09/inheritance.py
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
```

**Key Points:**
- `MISSING` marks required fields that must be overridden
- Child classes inherit all fields from the parent
- Child classes can add new fields (`batch_size`, `lr_scheduler`)
- Child classes can override parent defaults (`model`)

### 9.6 Defaults List in Structured Configs

You can define a defaults list directly in Python to specify which config group options to load.

```python
#structured_09/inheritance.py (continued)

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


@hydra.main(config_path=None, config_name="list_config", version_base=None)
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    main()
```

Running this:

```bash
python structured_09/inheritance.py
```

Output:

```bash
experiment:
  model: resnet18
  nrof_epochs: 30
  lr: 0.001
  batch_size: 128
```

**Overriding the default:**

```bash
python structured_09/inheritance.py experiment=resnet50
```

Output:

```bash
experiment:
  model: resnet50
  nrof_epochs: 30
  lr: 0.001
  lr_scheduler: cosine
```

### 9.7 Schema Validation with Pydantic

For advanced validation, you can use Pydantic dataclasses with field validators.

```python
#structured_09/validation.py
from typing import Any, List, Optional
from pydantic.dataclasses import dataclass
from pydantic import field_validator

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig, MISSING
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
    OmegaConf.to_object(config)  # This triggers validation
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    main()
```

**Key Points:**
- Use `from pydantic.dataclasses import dataclass` instead of standard dataclass
- `@field_validator` defines custom validation logic
- `OmegaConf.to_object(config)` converts DictConfig to the dataclass, triggering validation

### 9.8 Combining YAML and Structured Configs

The most powerful approach combines YAML files with structured configs for validation and defaults.

**YAML config that references a schema:**

```yaml
#structured_09/configs/config.yaml
defaults:
- config_schema
- experiment: resnet18
- _self_
```

```yaml
#structured_09/configs/experiment/resnet18.yaml
defaults:
  - resnet18_schema

model: resnet18
nrof_epochs: 30
lr: 0.001
batch_size: 16
```

```yaml
#structured_09/configs/experiment/resnet50.yaml
defaults:
  - resnet50_schema

model: resnet50
nrof_epochs: 30
lr: 0.001
batch_size: 64
lr_scheduler: cosine
```

**How it works:**
1. YAML files define the actual config values
2. Structured configs (schemas) define the types and validation
3. The YAML references the schema via defaults

Running with valid batch_size:

```bash
python structured_09/validation.py
```

Output:

```bash
experiment:
  model: resnet18
  nrof_epochs: 30
  lr: 0.001
  batch_size: 16
```

Running with resnet50:

```bash
python structured_09/validation.py experiment=resnet50
```

Output:

```bash
experiment:
  model: resnet50
  nrof_epochs: 30
  lr: 0.001
  batch_size: 64
  lr_scheduler: cosine
```

**Testing validation with invalid batch_size:**

```bash
python structured_09/validation.py experiment.batch_size=10
```

Output:

```bash
Error executing job with overrides: ['experiment.batch_size=10']
pydantic_core._pydantic_core.ValidationError: 1 validation error for Resnet18ExperimentSchema
batch_size
  Value error, Batch size must be divisible by 8 [type=value_error, input_value=10, input_type=int]
```

### 9.9 Using MISSING for Required Fields

`MISSING` is a special sentinel value that marks fields as required. If not provided, Hydra will raise an error.

```python
from omegaconf import MISSING

@dataclass
class DatabaseConfig:
    host: str = MISSING      # Required - must be provided
    port: int = 5432         # Optional - has default
    username: str = MISSING  # Required
    password: str = MISSING  # Required
```

**Best Practice:** Use `MISSING` for:
- Sensitive data (passwords, API keys)
- Environment-specific values (hosts, paths)
- Values that should never have a "default"

### 9.10 Optional Fields and None Handling

Use `Optional` from typing for fields that can be `None`.

```python
#structured_09/optional_fields.py
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
```

**Running with defaults (all Optional fields are None):**

```bash
python structured_09/optional_fields.py
```

Output:

```bash
training:
  model: resnet18
  epochs: 100
  lr: 0.001
  pretrained_weights: null
  lr_scheduler: null
  early_stopping_patience: null
  resume_from_checkpoint: null


--- Checking Optional Fields ---
No pretrained weights specified, training from scratch
No learning rate scheduler
Early stopping disabled
```

**Overriding Optional fields from CLI:**

```bash
python structured_09/optional_fields.py training.pretrained_weights=/path/to/weights.pth training.lr_scheduler=cosine
```

Output:

```bash
training:
  model: resnet18
  epochs: 100
  lr: 0.001
  pretrained_weights: /path/to/weights.pth
  lr_scheduler: cosine
  early_stopping_patience: null
  resume_from_checkpoint: null


--- Checking Optional Fields ---
Loading pretrained weights from: /path/to/weights.pth
Using learning rate scheduler: cosine
Early stopping disabled
```

### 9.11 Default Factories for Lists and Dicts

For mutable default values (lists, dicts), use `field(default_factory=...)`.

```python
#structured_09/lists_dicts.py
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
```

**Running with defaults:**

```bash
python structured_09/lists_dicts.py
```

Output:

```bash
data:
  batch_size: 32
  num_workers: 4
  train_datasets: []
  dataset_weights: {}
  augmentation:
    transforms:
    - resize
    - normalize
    transform_params: {}


--- Working with Lists and Dicts ---
Transforms to apply: ['resize', 'normalize']
Number of training datasets: 0
No dataset weights specified (equal weighting)
```

**Overriding lists and dicts from CLI:**

```bash
python structured_09/lists_dicts.py '+data.train_datasets=[imagenet,coco]' '+data.dataset_weights={imagenet:0.7,coco:0.3}'
```

Output:

```bash
data:
  batch_size: 32
  num_workers: 4
  train_datasets:
  - imagenet
  - coco
  dataset_weights:
    imagenet: 0.7
    coco: 0.3
  augmentation:
    transforms:
    - resize
    - normalize
    transform_params: {}


--- Working with Lists and Dicts ---
Transforms to apply: ['resize', 'normalize']
Number of training datasets: 2
Dataset weights: {'imagenet': 0.7, 'coco': 0.3}
```

**Important:** Use the `+` prefix when adding values to empty lists/dicts in structured configs.

**Why not use `= []` or `= {}`?**

Using mutable defaults directly causes all instances to share the same object:

```python
# DON'T DO THIS
@dataclass
class BadConfig:
    items: List[str] = []  # All instances share this list!
```

### 9.12 OmegaConf Interpolation with Structured Configs

You can use OmegaConf interpolation syntax in your YAML files even when using structured configs.

```python
#structured_09/interpolation.py
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
```

**YAML config with interpolations:**

```yaml
#structured_09/configs/interpolation_config.yaml
defaults:
  - base_config
  - _self_

# Use interpolation to build paths dynamically
output_dir: outputs/${experiment.model}_${experiment.dataset}/${now:%Y-%m-%d_%H-%M-%S}
checkpoint_dir: ${output_dir}/checkpoints
log_file: ${output_dir}/train.log
```

**Running with defaults:**

```bash
python structured_09/interpolation.py
```

Output:

```bash
experiment:
  model: resnet18
  dataset: imagenet
  epochs: 100
  lr: 0.001
output_dir:
outputs/${experiment.model}_${experiment.dataset}/${now:%Y-%m-%d_%H-%M-%S}
checkpoint_dir: ${output_dir}/checkpoints
log_file: ${output_dir}/train.log


--- Resolved Paths ---
Output directory: outputs/resnet18_imagenet/2025-11-20_06-14-32
Checkpoint directory: outputs/resnet18_imagenet/2025-11-20_06-14-32/checkpoints
Log file: outputs/resnet18_imagenet/2025-11-20_06-14-32/train.log
```

**Overriding experiment values (interpolations update automatically):**

```bash
python structured_09/interpolation.py experiment.model=resnet50 experiment.dataset=coco
```

Output:

```bash
experiment:
  model: resnet50
  dataset: coco
  epochs: 100
  lr: 0.001
output_dir:
outputs/${experiment.model}_${experiment.dataset}/${now:%Y-%m-%d_%H-%M-%S}
checkpoint_dir: ${output_dir}/checkpoints
log_file: ${output_dir}/train.log


--- Resolved Paths ---
Output directory: outputs/resnet50_coco/2025-11-20_06-14-36
Checkpoint directory: outputs/resnet50_coco/2025-11-20_06-14-36/checkpoints
Log file: outputs/resnet50_coco/2025-11-20_06-14-36/train.log
```

**Key Points:**
- The structured config defines the field types and defaults
- The YAML file uses interpolation syntax: `${field.name}` and `${now:%format}`
- Interpolations are resolved at runtime
- Changing interpolated values automatically updates dependent fields

### 9.13 Using _target_ for Object Instantiation

Combine structured configs with Hydra's `instantiate` API for automatic object creation.

```python
from dataclasses import dataclass
from hydra.utils import instantiate

@dataclass
class OptimizerConfig:
    _target_: str = "torch.optim.Adam"
    lr: float = 0.001
    weight_decay: float = 0.0001

@dataclass
class Config:
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

# In your main function:
@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    model = MyModel()
    optimizer = instantiate(config.optimizer, params=model.parameters())
    # optimizer is now a torch.optim.Adam instance
```

### 9.14 Common Patterns

#### Pattern 1: Base Config with Environment-Specific Overrides

```python
@dataclass
class BaseConfig:
    debug: bool = False
    log_level: str = "INFO"

@dataclass
class DevConfig(BaseConfig):
    debug: bool = True
    log_level: str = "DEBUG"

@dataclass
class ProdConfig(BaseConfig):
    debug: bool = False
    log_level: str = "WARNING"

cs.store(group="env", name="dev", node=DevConfig)
cs.store(group="env", name="prod", node=ProdConfig)
```

#### Pattern 2: Composable Model Configs

```python
@dataclass
class BackboneConfig:
    _target_: str = MISSING
    pretrained: bool = True

@dataclass
class ResNet18Backbone(BackboneConfig):
    _target_: str = "torchvision.models.resnet18"

@dataclass
class ResNet50Backbone(BackboneConfig):
    _target_: str = "torchvision.models.resnet50"

@dataclass
class HeadConfig:
    _target_: str = MISSING
    num_classes: int = MISSING

@dataclass
class ClassificationHead(HeadConfig):
    _target_: str = "my_models.ClassificationHead"
    dropout: float = 0.5

@dataclass
class ModelConfig:
    backbone: BackboneConfig = field(default_factory=ResNet18Backbone)
    head: HeadConfig = field(default_factory=ClassificationHead)
```

#### Pattern 3: Config with Computed Defaults

```python
@dataclass
class DataConfig:
    batch_size: int = 32
    num_workers: int = 4

    def __post_init__(self):
        # Computed default based on other fields
        if not hasattr(self, 'prefetch_factor'):
            self.prefetch_factor = 2 if self.num_workers > 0 else None
```

### 9.15 Common Pitfalls

#### Pitfall 1: Forgetting to Use default_factory for Nested Dataclasses

**Problem:**

```python
@dataclass
class Config:
    experiment: ExperimentConfig = ExperimentConfig()  # WRONG!
```

This creates a single shared instance.

**Solution:**

```python
@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
```

#### Pitfall 2: Using Standard Dataclass with Pydantic Validators

**Problem:**

```python
from dataclasses import dataclass  # Standard dataclass
from pydantic import field_validator

@dataclass
class Config:
    batch_size: int = 32

    @field_validator("batch_size")  # Won't work!
    def validate(cls, v):
        return v
```

**Solution:**

```python
from pydantic.dataclasses import dataclass  # Pydantic dataclass
```

#### Pitfall 3: Not Calling OmegaConf.to_object() for Validation

**Problem:**

```python
@hydra.main(...)
def main(config: DictConfig):
    # Validation never runs because config is still a DictConfig
    print(config.batch_size)  # No validation error even if invalid
```

**Solution:**

```python
@hydra.main(...)
def main(config: DictConfig):
    OmegaConf.to_object(config)  # Triggers validation
    print(config.batch_size)
```

#### Pitfall 4: Mixing Type Annotations Incorrectly

**Problem:**

```python
@dataclass
class Config:
    experiment: Resnet18Experiment  # Too specific!
```

This forces only `Resnet18Experiment` to be used.

**Solution:**

```python
from typing import Any

@dataclass
class Config:
    experiment: Any  # Allows any config group option
```

Or use a base class:

```python
@dataclass
class Config:
    experiment: Experiment  # Allows any subclass of Experiment
```

#### Pitfall 5: Forgetting _self_ in Defaults

**Problem:**

```python
DEFAULT = [
    {"experiment": "resnet18"},
    # Missing _self_!
]
```

Without `_self_`, the current file's content won't be merged.

**Solution:**

```python
DEFAULT = [
    {"experiment": "resnet18"},
    "_self_",
]
```

#### Pitfall 6: Circular Imports with ConfigStore

**Problem:**

```python
# config_a.py
from config_b import ConfigB  # Circular!

# config_b.py
from config_a import ConfigA  # Circular!
```

**Solution:**

Use a central config module or lazy imports:

```python
# configs/__init__.py
from .experiment import ExperimentConfig
from .loss import LossConfig
from .main import Config

# Register all configs here
cs = ConfigStore.instance()
cs.store(name="config", node=Config)
# ...
```

#### Pitfall 7: Schema Mismatch Between YAML and Dataclass

**Problem:**

```yaml
# config.yaml
experiment:
  model: resnet18
  extra_field: value  # Not in dataclass!
```

```python
@dataclass
class ExperimentConfig:
    model: str = "resnet18"
    # No extra_field defined
```

OmegaConf will accept it, but `to_object()` will fail.

**Solution:**

Either add the field to the dataclass or use `struct` mode:

```python
OmegaConf.set_struct(config, True)  # Raises error on unknown fields
```
