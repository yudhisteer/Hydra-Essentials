# Hydra-Essentials

------------------------------------------------------------------------------------------------------------------------------------------

## 1. CLI Commands

### 1.1 Config using CLI

```bash
#scripts/01_hydra.py
from omegaconf import OmegaConf, DictConfig
import hydra

from rich import print

import warnings
warnings.filterwarnings("ignore")


# we do nt have a config file yet
@hydra.main(config_path=None)
def main(config: DictConfig) -> None:
    print("Printing config...")
    print(OmegaConf.to_yaml(config))  # we are convetng that config to a yaml string. Will return "{}" 


if __name__ == "__main__":
    main()
```

Since we have no config file, the file will print "{}". However, we can specify configs in the terminal using the Hydra CLI asu such:
Noticw that we use a single "+" to specify the config.

```bash
python 'scripts/01_hydra.py' +training.batch_size=128 +training.nrof_epochs=30 +training.lr=5e-3                                                      
Printing config...
training:
  batch_size: 128
  nrof_epochs: 30
  lr: 0.005
```

### 1.2 Config using Config File

Now let's creeate a config file and use it in the script.

```yaml
#scripts/config.yaml
training:
  batch_size: 128
  nrof_epochs: 30
  lr: 5e-3
```

We update our script to specify the location of the config file and its name.

```bash
#scripts/01_hydra.py
from omegaconf import OmegaConf, DictConfig
import hydra

from rich import print

import warnings
warnings.filterwarnings("ignore")


# we have a config file now
# we specify the location of it and its name
@hydra.main(config_path=".", config_name="config.yaml")
def main(config: DictConfig) -> None:
    print("Printing config...")
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    main()
```

This will print:

```bash
Printing config...
training:
  batch_size: 128
  nrof_epochs: 30
  lr: 0.005
```

If we run the  script now, it will print the config file. However, we can also update the config in the terminal as such:
Note: We now use two "+" signs to update the config as it already exists in the config file.

```bash
python 'scripts/01_hydra.py' ++training.batch_size=256
```

This will update the config file and print the new config.

```bash
Printing config...
training:
  batch_size: 256
  nrof_epochs: 60
  lr: 0.005
```

We can also create new configs using the two "+" signs. This will overwrite your original config file.

```bash
python 'scripts/01_hydra.py' ++training.batch_size=256 ++model.name=resnet18
```

This will print:

```bash
Printing config...
training:
  batch_size: 256
  nrof_epochs: 60
  lr: 0.001
model:
  name: resnet18
```

When you run any of the script above, it will creeate folder in the "outputs" folder with the name as the timestamp. In the ".hydra" folder, you will find the `config.yaml` file, the `overrides.yaml` file, and the `hydra.yaml` file.

```yaml
#scripts/config.yaml
training:
  batch_size: 256
  nrof_epochs: 30
  lr: 0.005
model:
  name: resnet18
```

```yaml
#scripts/overrides.yaml
- ++training.batch_size=256
- ++model.name=resnet18
```

----------------------------------------------------------------------------------

## 2. OmegaConf

OmegaConf is a Python library for configuration management. Hydra's configuration system is built on top of OmegaConf, providing more advanced features which we will explore later.

**Why should you use OmegaConf?**
- Hierarchical configurations: Organize configs with nested structures
- Type safety: Has runtime type validation with structured configs using dataclasses
- Variable interpolation: Reference other config values with `${key.sub}`
- Config merging: Combine multiple configs
- Missing value validation: Mark mandatory fields with `???`
- Read-only and struct modes: Prevent accidental modifications
- Environment variables: Inject values from environment

**How it works:**
OmegaConf wraps Python dicts and lists in `DictConfig` and `ListConfig` objects, providing additional functionality like `interpolation`, `validation`, and `type checking`.

**Note: All examples in this section are located in the `omegaconf_02/` directory.**

### 2.1 Creating Configs

#### Creating from Python Objects

OmegaConf can create configs from Python dicts, lists, and even dataclasses.

```python
#omegaconf_02/01_creating_configs.py
from omegaconf import OmegaConf

# From dict
config = OmegaConf.create({
    "training": {
        "batch_size": 128,
        "epochs": 30,
        "lr": 0.001
    },
    "model": {
        "name": "resnet18",
        "pretrained": True
    }
})
print(OmegaConf.to_yaml(config))
```

Output:
```yaml
training:
  batch_size: 128
  epochs: 30
  lr: 0.001
model:
  name: resnet18
  pretrained: true
```

#### Creating ListConfigs

```python
# From list
config_list = OmegaConf.create([1, 2, 3, "four", {"key": "value"}])
print(OmegaConf.to_yaml(config_list))
```

Output:
```yaml
- 1
- 2
- 3
- four
- key: value
```

#### Loading from Files

```python
# Load from YAML file
config = OmegaConf.load("omegaconf_02/configs/training.yaml")
```

```yaml
#omegaconf_02/configs/training.yaml
training:
  batch_size: 128
  epochs: 30
  lr: 0.001
  optimizer: adam
  scheduler: cosine

model:
  name: resnet18
  pretrained: true
```

#### Creating from Dotlist

```python
# From list of dot-separated strings
dotlist = ["training.batch_size=256", "training.lr=0.0001", "model.name=resnet50"]
config_from_dotlist = OmegaConf.from_dotlist(dotlist)
```

Output:
```yaml
training:
  batch_size: 256
  lr: 0.0001
model:
  name: resnet50
```

You can try the command above by running:

**Test command:**
```bash
python omegaconf_02/01_creating_configs.py
```

### 2.2 Accessing and Modifying Configs

#### Accessing Values

OmegaConf provides multiple ways to access configuration values:

```python
#omegaconf_02/02_accessing_modifying.py
config = OmegaConf.load("omegaconf_02/configs/training.yaml")
```

```yaml
#omegaconf_02/configs/training.yaml
training:
  batch_size: 128
  epochs: 30
  lr: 0.001
  optimizer: adam
  scheduler: cosine

model:
  name: resnet18
  pretrained: true
```

```python
#omegaconf_02/02_accessing_modifying.py
# Dot notation
batch_size = config.training.batch_size  # 128
model_name = config.model.name  # "resnet18"

# Bracket notation
lr = config["training"]["lr"]  # 0.001
pretrained = config["model"]["pretrained"]  # True
```

#### Safe Access with OmegaConf.select()

Use `select()` to safely access keys that might not exist:

```python
# Returns None if key doesn't exist (instead of raising error)
momentum = OmegaConf.select(config, "training.momentum", default=0.9)
print(f"Momentum: {momentum}")  # 0.9

weight_decay = OmegaConf.select(config, "training.weight_decay")
print(f"Weight decay: {weight_decay}")  # None
```

#### Modifying Values

```python
# Direct assignment
config.training.batch_size = 256
config.training.lr = 0.0001

# Adding new keys (when struct mode is disabled)
config.training.weight_decay = 0.0001
config.new_section = {"key": "value"}
```

Output after modifications:
```yaml
training:
  batch_size: 256
  epochs: 30
  lr: 0.0001
  optimizer: adam
  scheduler: cosine
  weight_decay: 0.0001
model:
  name: resnet18
  pretrained: true
new_section:
  key: value
```

**Test command:**
```bash
python omegaconf_02/02_accessing_modifying.py
```

### 2.3 Variable Interpolation

#### Basic Interpolation

Reference other config values using `${key.subkey}` syntax:

```yaml
#omegaconf_02/configs/interpolation.yaml
server:
  host: localhost
  port: 8080
  url: ${server.host}:${server.port}

client:
  server_url: ${server.url}
  timeout: 30

paths:
  root: /data
  train: ${paths.root}/train
  test: ${paths.root}/test
```

```python
#omegaconf_02/03_interpolation.py
config = OmegaConf.load("omegaconf_02/configs/interpolation.yaml")

# Without resolve - shows interpolation syntax
print(OmegaConf.to_yaml(config))
```

Output without resolve:
```yaml
server:
  host: localhost
  port: 8080
  url: ${server.host}:${server.port}
client:
  server_url: ${server.url}
  timeout: 30
paths:
  root: /data
  train: ${paths.root}/train
  test: ${paths.root}/test
```

```python
# With resolve - shows final values
print(OmegaConf.to_yaml(config, resolve=True))
```

Output with resolve:
```yaml
server:
  host: localhost
  port: 8080
  url: localhost:8080
client:
  server_url: localhost:8080
  timeout: 30
paths:
  root: /data
  train: /data/train
  test: /data/test
```

**Note:** OmegaConf automatically resolves interpolations when you access values:
```python
print(config.client.server_url)  # "localhost:8080" (already resolved)
```

#### Relative Interpolation

Use `${.key}` to reference sibling keys:

```python
relative_config = OmegaConf.create({
    "database": {
        "host": "localhost",
        "port": 5432,
        "connection": "${.host}:${.port}"  # ${.} references siblings
    }
})
```

Output:
```yaml
database:
  host: localhost
  port: 5432
  connection: localhost:5432
```

#### Environment Variable Interpolation

Access environment variables with `${oc.env:VAR_NAME}`:

```python
import os
os.environ["APP_NAME"] = "MyApp"
os.environ["VERSION"] = "1.0.0"

env_config = OmegaConf.create({
    "app": {
        "name": "${oc.env:APP_NAME}",
        "version": "${oc.env:VERSION}",
        "debug": "${oc.env:DEBUG,false}"  # Default value if not set
    }
})
```

Output:
```yaml
app:
  name: MyApp
  version: 1.0.0
  debug: 'False'
```

**Test command:**
```bash
python omegaconf_02/03_interpolation.py
```

### 2.4 Missing Values and Validation

#### Mandatory Values with ???

Use `???` to mark values as mandatory. Accessing them before assignment raises an error:

```yaml
#omegaconf_02/configs/missing_values.yaml
database:
  host: localhost
  port: 5432
  username: ???  # Mandatory field
  password: ???  # Mandatory field
```

```python
#omegaconf_02/04_missing_values.py
from omegaconf import OmegaConf, MissingMandatoryValue

config = OmegaConf.load("omegaconf_02/configs/missing_values.yaml")

# Check if value is missing
is_missing = OmegaConf.is_missing(config, "database.username")
print(f"Is username missing? {is_missing}")  # False (??? is not considered missing by is_missing)

# Attempting to access missing value
try:
    username = config.database.username
except MissingMandatoryValue as e:
    print(f"Error: {e}")
```

Here, we see that ??? is not considered missing by `OmegaConf.is_missing()`. However, if we try to access the missing value, it will raise an error.

Output:
```
Error: Missing mandatory value: database.username
    full_key: database.username
    object_type=dict
```

#### Setting Mandatory Values

```python
# Must set values before use
config.database.username = "admin"
config.database.password = "secret123"

# Now accessing works
print(config.database.username)  # "admin"
```

**Test command:**
```bash
python omegaconf_02/04_missing_values.py
```

### 2.5 Merging Configurations

#### OmegaConf.merge() - Returns New Config

`merge()` creates a new config without modifying the originals:

```python
#omegaconf_02/05_merging.py
base = OmegaConf.load("omegaconf_02/configs/base.yaml")
override = OmegaConf.load("omegaconf_02/configs/override.yaml")
```

```yaml
#omegaconf_02/configs/base.yaml
model:
  name: resnet18
  layers: 18
  pretrained: false

training:
  batch_size: 32
  lr: 0.001
```

```yaml
#omegaconf_02/configs/override.yaml
model:
  pretrained: true
  dropout: 0.5

training:
  batch_size: 128
  epochs: 100

optimizer:
  type: adam
  weight_decay: 0.0001
```


```python
# Merge returns new config
merged = OmegaConf.merge(base, override)

# Original configs unchanged
print(f"Base batch_size: {base.training.batch_size}")  # 32
print(f"Merged batch_size: {merged.training.batch_size}")  # 128
```

However, we now have a new config `merged` which is the merged of the `base` and `override` configs.

```yaml
Merged config:
model:
  name: resnet18
  layers: 18
  pretrained: true
  dropout: 0.5
training:
  batch_size: 128
  lr: 0.001
  epochs: 100
optimizer:
  type: adam
  weight_decay: 0.0001
```

#### Merge Behavior

Later configs override earlier ones:

```python
config1 = OmegaConf.create({"a": 1, "b": 2})
config2 = OmegaConf.create({"b": 3, "c": 4})
config3 = OmegaConf.create({"c": 5, "d": 6})

merged = OmegaConf.merge(config1, config2, config3)
```

Output:
```yaml
a: 1     # from config1
b: 3     # from config2 (overrides config1)
c: 5     # from config3 (overrides config2)
d: 6     # from config3
```

#### OmegaConf.update() - In-Place Modification

`update()` modifies config in-place:

```python
OmegaConf.update(config, "training.batch_size", 512)
OmegaConf.update(config, "model.pretrained", True)
```

```yaml
model:
  name: resnet18
  layers: 18
  pretrained: true
training:
  batch_size: 512
  lr: 0.001
```

**Test command:**
```bash
python omegaconf_02/05_merging.py
```

### 2.6 Struct Mode and Read-Only Configs

#### Struct Mode - Prevents Adding New Keys

By default, you can add new keys to configs. Struct mode prevents this to catch typos. However, you can still modify existing keys.

```python
#omegaconf_02/06_struct_mode.py
config = OmegaConf.create({"model": {"name": "resnet18"}})
OmegaConf.set_struct(config, True)

# Modifying existing key works
config.model.name = "resnet50"  # OK

# Adding new key raises error
try:
    config.model.layers = 18
except Exception as e:
    print(f"Error: {e}")
```

Output:
```
Error: Key 'layers' is not in struct
    full_key: model.layers
    object_type=dict
```

#### Read-Only Mode - Prevents All Modifications

You can set a config to be read-only to prevent any modifications to the config such as adding new keys or modifying existing keys.

```python
readonly_config = OmegaConf.create({"model": {"name": "resnet18"}})
OmegaConf.set_readonly(readonly_config, True)

# Any modification raises error
try:
    readonly_config.model.name = "resnet50"
except Exception as e:
    print(f"Error: {e}")
```

Output:
```
Error: Cannot change read-only config container
    full_key: model.name
    object_type=dict
```

#### Converting to Python Dict

Use `to_container()` to convert OmegaConf objects to plain Python dict/list:

```python
config = OmegaConf.create({
    "model": {"name": "resnet18", "layers": 18},
    "training": {"batch_size": 128}
})

python_dict = OmegaConf.to_container(config)
print(type(python_dict))  # <class 'dict'>

# Now it's a regular dict
python_dict["new_key"] = "works"
```

```python
 {'model': {'name': 'resnet18', 'layers': 18}, 'training': {'batch_size': 128}, 'new_key': 'works'}
```

**Test command:**
```bash
python omegaconf_02/06_struct_mode.py
```

### 2.7 Structured Configs (Dataclasses)

Structured configs use Python dataclasses for type safety and validation:

```python
#omegaconf_02/07_structured_configs.py
from dataclasses import dataclass, field
from omegaconf import OmegaConf

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

# Create config from dataclass
config = OmegaConf.structured(Config)
```

Output:
```yaml
model:
  name: resnet18
  layers: 18
  pretrained: false
  dropout: 0.5
training:
  batch_size: 32
  epochs: 100
  lr: 0.001
  optimizer: adam
seed: 42
```

#### Type Safety

Structured configs enforce types at runtime:

```python
# Correct type - works
config.training.batch_size = 128

# Wrong type - raises error
try:
    config.training.batch_size = "invalid"  # Should be int
except Exception as e:
    print(f"Error: {e}")
```

Output:
```
Error: Value 'invalid' of type 'str' could not be converted to Integer
    full_key: training.batch_size
    reference_type=TrainingConfig
    object_type=TrainingConfig
```

#### Merging with Unstructured Configs

```python
yaml_config = OmegaConf.create({
    "model": {"pretrained": True, "layers": 50},
    "training": {"epochs": 200}
})

merged = OmegaConf.merge(config, yaml_config)
```

Output (types are validated during merge):
```yaml
model:
  name: resnet18
  layers: 50
  pretrained: true
  dropout: 0.5
training:
  batch_size: 128
  epochs: 200
  lr: 0.001
  optimizer: adam
seed: 42
```

Note: Structured configs behave as if struct mode is enabled, so we cannot add new fields to the config object.

```python
try:
    merged.new_field = "value"
except Exception as e:
    print(f"Error when adding new field: {type(e).__name__}: {e}")
```

Output:
```
Error when adding new field: ConfigAttributeError: Key 
'new_field' not in 'Config'
    full_key: new_field
    object_type=Config
```

#### Converting Back to Python Object

```python
config_obj = OmegaConf.to_object(merged)
print(type(config_obj))  # <class '__main__.Config'>
print(config_obj.model.name)  # "resnet18"
```

**Test command:**
```bash
python omegaconf_02/07_structured_configs.py
```

### 2.8 Working with Lists

#### Creating and Using ListConfigs

```python
#omegaconf_02/08_lists.py
list_config = OmegaConf.create([1, 2, 3, 4, 5])

# Indexing
print(list_config[0])  # 1
print(list_config[-1])  # 5

# Slicing
print(list_config[1:3])  # [2, 3]

# Modifying
list_config[0] = 10
list_config.append(6)
```

Output:
```yaml
- 10
- 2
- 3
- 4
- 5
- 6
```

#### Lists in Configs

```python
config = OmegaConf.create({
    "model": {
        "layers": [64, 128, 256, 512],
        "dropout_rates": [0.1, 0.2, 0.3]
    },
    "data": {
        "transforms": ["resize", "normalize", "augment"]
    }
})

# Accessing
print(config.model.layers[0])  # 64
print(config.data.transforms)  # ['resize', 'normalize', 'augment']
```

#### List Interpolation

```python
interp_config = OmegaConf.create({
    "base_layers": [64, 128],
    "model": {
        "encoder_layers": "${base_layers}",
        "decoder_layers": [256, 512]
    }
})
```

With resolve:
```yaml
base_layers:
- 64
- 128
model:
  encoder_layers:
  - 64
  - 128
  decoder_layers:
  - 256
  - 512
```

**Test command:**
```bash
python omegaconf_02/08_lists.py
```

### 2.9 CLI Integration

#### Parsing Command Line Arguments

```python
#omegaconf_02/09_cli.py
from omegaconf import OmegaConf
import sys

# Parse CLI arguments
config = OmegaConf.from_cli()

# Load base config and merge with CLI
base = OmegaConf.load("omegaconf_02/configs/training.yaml")
merged = OmegaConf.merge(base, config)
```

```yaml
#omegaconf_02/configs/training.yaml
training:
  batch_size: 128
  epochs: 30
  lr: 0.001
  optimizer: adam
  scheduler: cosine

model:
  name: resnet18
  pretrained: true
```

**Test command:**
```bash
python omegaconf_02/09_cli.py training.batch_size=256 model.name=resnet50 training.lr=0.0001
```

Output:
```
Command line args: ['training.batch_size=256', 'model.name=resnet50', 'training.lr=0.0001']

Parsed configuration:
training:
  batch_size: 256
  lr: 0.0001
model:
  name: resnet50

Merged result:
training:
  batch_size: 256
  epochs: 30
  lr: 0.0001
  optimizer: adam
  scheduler: cosine
model:
  name: resnet50
  pretrained: true
```

### 2.10 Saving and Loading Configs

#### Saving Configs to File

```python
#omegaconf_02/10_saving.py
config = OmegaConf.create({
    "model": {"name": "resnet50", "pretrained": True},
    "training": {"batch_size": 128, "epochs": 100}
})

# Save to file
OmegaConf.save(config, "saved_config.yaml")

# Load from file
loaded = OmegaConf.load("saved_config.yaml")

# Verify they're equal
print(config == loaded)  # True
```

**Test command:**
```bash
python omegaconf_02/10_saving.py
```

### 2.11 Common Use Cases

#### Use Case 1: Configuration Inheritance

```python
# Base configuration for all experiments
base_config = OmegaConf.create({
    "model": {"pretrained": False},
    "training": {"batch_size": 32, "epochs": 100}
})

# Specific experiment overrides
experiment_config = OmegaConf.create({
    "model": {"pretrained": True},
    "training": {"batch_size": 128}
})

# Merge for final config
config = OmegaConf.merge(base_config, experiment_config)
```

#### Use Case 2: Dynamic Path Generation

```python
config = OmegaConf.create({
    "experiment_name": "resnet50_imagenet",
    "output_dir": "/outputs",
    "paths": {
        "checkpoints": "${output_dir}/${experiment_name}/checkpoints",
        "logs": "${output_dir}/${experiment_name}/logs",
        "results": "${output_dir}/${experiment_name}/results"
    }
})
```

Resolved paths:
```
checkpoints: /outputs/resnet50_imagenet/checkpoints
logs: /outputs/resnet50_imagenet/logs
results: /outputs/resnet50_imagenet/results
```

#### Use Case 3: Type-Safe Configuration

```python
@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    max_connections: int = 10

config = OmegaConf.structured(DatabaseConfig)

# This is caught at runtime
try:
    config.port = "invalid"  # Should be int
except ValidationError:
    print("Type error caught!")
```

### 2.12 Best Practices and Common Pitfalls

#### Best Practices

**1. Use structured configs for type safety**

```python
# Good - type safe
@dataclass
class Config:
    batch_size: int = 32
    lr: float = 0.001

# vs Basic dict config (no type checking)
config = {"batch_size": 32, "lr": 0.001}
```

**2. Use OmegaConf.select() for optional keys**

```python
# Good - won't raise error if missing
value = OmegaConf.select(config, "optional.key", default=None)

# Bad - raises error if missing
value = config.optional.key
```

**3. Use struct mode to catch typos**

```python
OmegaConf.set_struct(config, True)
# Typos in key names will raise errors instead of silently creating new keys
```

**4. Use relative interpolation for related values**

```python
# Good - uses relative reference
server:
  host: localhost
  port: 8080
  url: ${.host}:${.port}

# vs Absolute reference (less maintainable)
server:
  host: localhost
  port: 8080
  url: ${server.host}:${server.port}
```

**5. Validate mandatory fields with ???**

```python
database:
  username: ???  # Forces explicit configuration
  password: ???
```

#### Common Pitfalls

**Pitfall 1: Forgetting to resolve interpolations**

```python
# Problem: Seeing ${...} in outputs
config = OmegaConf.load("config.yaml")
print(OmegaConf.to_yaml(config))  # Shows ${...}

# Solution: Use resolve=True
print(OmegaConf.to_yaml(config, resolve=True))  # Shows actual values
```

**Pitfall 2: Modifying shared configs**

```python
# Problem: Modifications affect all references
base = OmegaConf.create({"a": 1})
config1 = base
config2 = base
config1.a = 2
print(config2.a)  # 2 (unexpected!)

# Solution: Use merge to create copies
config1 = OmegaConf.merge(base, {})
config2 = OmegaConf.merge(base, {})
```

**Pitfall 3: Accessing ??? values without checking**

```python
# Problem: Crashes at runtime
config = OmegaConf.create({"password": "???"})
pwd = config.password  # MissingMandatoryValue error!

# Solution: Check first or set value
if OmegaConf.is_missing(config, "password"):
    config.password = get_password()
```

**Pitfall 4: Wrong interpolation syntax**

```python
# Wrong - single braces
config: "{key}"  # Treated as literal string.

# Wrong - missing oc. prefix for env vars
env_var: "${ENV_VAR}"  # Won't work.
# Correct
config: "${key}"
env_var: "${oc.env:ENV_VAR}"
```

**Pitfall 5: Type coercion confusion**

```python
# Strings are coerced to numbers when possible
config = OmegaConf.create({"value": "123"})
print(type(config.value))  # str

# But in structured configs, strict validation applies
@dataclass
class Config:
    value: int

config = OmegaConf.structured(Config)
config.value = "123"  # Error! String cannot be assigned to int field
```
-------------------------------------------------------------------------

## 3. Config Groups

Hydra's config groups allow you to organize related configurations into groups and easily switch between them. This is fundamental for managing different experiments, models, datasets, or any other configuration variants.

**Why use config groups?**
- **Organization**: Keep related configs together (e.g., all model configs in `configs/model/`)
- **Reusability**: Define configs once, use them across multiple experiments
- **Flexibility**: Switch between configurations easily via CLI or defaults
- **Maintainability**: Easier to manage large projects with many configuration variants
- **Composition**: Combine multiple config groups to create complete configurations

**How it works:**
Config groups are organized in subdirectories under your config path. The subdirectory name becomes the group name, and each YAML file in that directory is an option for that group.

**All examples in this section are located in the `grouping_03/` directory.**

### 3.1 Basic Config Groups

We start by creating two experiment configuration files in the `experiment/` group:

```yaml
#grouping_03/configs/experiment/experiment_with_resnet18.yaml
model: resnet18
epochs: 100
batch_size: 128
lr: 0.001
optimizer: adam
scheduler: cosine
```

```yaml
#grouping_03/configs/experiment/experiment_with_resnet50.yaml
model: resnet50
epochs: 100
batch_size: 128
lr: 0.001
optimizer: adam
scheduler: cosine
```

**Understanding the Structure:**

```
grouping_03/
└── configs/
    ├── config.yaml           # Main config file
    └── experiment/           # Config group named "experiment"
        ├── experiment_with_resnet18.yaml  # Option 1
        └── experiment_with_resnet50.yaml  # Option 2
```

The directory name `experiment/` creates a config group called `experiment`. Each YAML file inside is an option you can select.

```python
#grouping_03/grouping.py
from omegaconf import OmegaConf, DictConfig
import hydra
from rich import print
import warnings
warnings.filterwarnings("ignore")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    main()
```

### 3.2 Adding Config Groups from CLI

With an empty `config.yaml`, we can add config groups from the command line using the `+` prefix: 

```yaml
#grouping_03/configs/config.yaml
# Empty config
```

Run it:

```bash
python grouping_03/grouping.py +experiment=experiment_with_resnet18 hydra.job.chdir=False
experiment:
  model: resnet18
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: adam
  scheduler: cosine
```

**What's happening:**
- The `+` prefix tells Hydra to **add** this config group to the composition
- `experiment=experiment_with_resnet18` selects the `experiment_with_resnet18.yaml` file from the `experiment/` group
- The content is placed under the `experiment` key in the final config

**CLI Syntax:**
- `+group=option` - Add a config group (errors if already in defaults)
- `group=option` - Override a config group that's already in defaults
- `~group` - Delete a config group from defaults

### 3.3 Setting Default Config Groups

Instead of specifying config groups every time from the CLI, we can set defaults in `config.yaml`:

```yaml
#grouping_03/configs/config_with_defaults.yaml
defaults:
  - experiment: experiment_with_resnet18
```

Now we can run without specifying the experiment:

```bash
python grouping_03/grouping.py --config-name=config_with_defaults hydra.job.chdir=False
experiment:
  model: resnet18
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: adam
  scheduler: cosine
```

**Benefits of defaults:**
- No need to specify config groups on every run
- Clear documentation of what configs are being used
- Can still override from CLI: `experiment=experiment_with_resnet50`

### 3.4 Overriding Default Config Groups

You can override default config groups in the defaults list:

```yaml
#grouping_03/configs/config_with_override.yaml
defaults:
  - experiment: experiment_with_resnet18
  - override experiment: experiment_with_resnet50  # This overrides the previous default
```

This will use ResNet50:

```bash
python grouping_03/grouping.py --config-name=config_with_override hydra.job.chdir=False
experiment:
  model: resnet50
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: adam
  scheduler: cosine
```

**Why use `override`?**
- Makes it explicit that you're overriding a previous default
- Prevents accidental duplication errors
- Better for readability in complex configs

### 3.5 Using `_self_` to Control Merge Order

The `_self_` keyword controls when the primary config's values are merged relative to defaults. This is crucial for understanding config composition:

```yaml
#grouping_03/configs/config_with_self.yaml
defaults:
  - experiment: experiment_with_resnet18
  - override experiment: experiment_with_resnet50
  - _self_

experiment:
  optimizer: SGD
```

The `_self_` determines when values from the primary config are applied:

```bash
python grouping_03/grouping.py --config-name=config_with_self hydra.job.chdir=False
experiment:
  model: resnet50
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: SGD      # ← This overrides the optimizer from the default
  scheduler: cosine
```

**Understanding `_self_`:**

```yaml
defaults:
  - experiment: experiment_with_resnet50  # Sets optimizer: adam
  - _self_                                 # Then apply primary config

experiment:
  optimizer: SGD  # This overrides adam because _self_ comes AFTER
```

**Order matters:**

```yaml
# _self_ at END (most common): Primary config overrides defaults
defaults:
  - experiment: experiment_with_resnet50
  - _self_
experiment:
  optimizer: SGD  # This wins (SGD)

# _self_ at START: Defaults override primary config
defaults:
  - _self_
  - experiment: experiment_with_resnet50
experiment:
  optimizer: SGD  # This gets overridden (result: adam)
```

**Best practice:** Place `_self_` at the end so your explicit config values take precedence.

### 3.6 Composing Multiple Config Groups

You can combine multiple config groups to build your complete configuration:

```yaml
#grouping_03/configs/demo_config.yaml
seed: 42
```

You can include standalone config files (not in a group directory) directly in the defaults list:

```yaml
#grouping_03/configs/config_with_merge.yaml
defaults:
  - experiment: experiment_with_resnet18
  - demo_config     # ← Standalone config file
  - _self_

experiment:
  optimizer: SGD
```

Running this will merge all configs:

```bash
python grouping_03/grouping.py --config-name=config_with_merge hydra.job.chdir=False
experiment:
  model: resnet18
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: SGD
  scheduler: cosine
seed: 42              # ← Merged from demo_config.yaml
```

**Composition Flow:**
1. Load `experiment/experiment_with_resnet18.yaml` → Sets experiment config
2. Load `demo_config.yaml` → Adds seed
3. Apply `_self_` → Overrides experiment.optimizer to SGD

**Understanding Group vs Non-Group Configs:**

```yaml
defaults:
  - experiment: experiment_with_resnet18  # Config GROUP: selects from experiment/ directory
  - demo_config                            # Standalone config: loads demo_config.yaml directly
```

- **Config Group** (`experiment: option`): Subdirectory with multiple options
  - Located in `configs/experiment/`
  - Requires `group: option` syntax
  - Content placed under `experiment` key by default

- **Standalone Config** (`demo_config`): Single YAML file in root
  - Located directly in `configs/`
  - Just the filename (no group)
  - Content merged at root level by default

### 3.7 Advanced Config Group Features

#### Optional Config Groups

Use the `optional` keyword to prevent errors if a config doesn't exist:

```yaml
defaults:
  - experiment: experiment_with_resnet18
  - optional dataset: imagenet  # Won't error if imagenet.yaml doesn't exist
  - optional custom_overrides   # Won't error if custom_overrides.yaml doesn't exist
```

**When to use:**
- User-specific configs that might not exist in all environments
- Optional plugins or extensions
- Development overrides that don't exist in production

**Example use case:**

```yaml
#configs/config.yaml
defaults:
  - model: resnet50
  - dataset: imagenet
  - optional local_overrides  # Developers can add local_overrides.yaml for personal settings
  - _self_
```

Each developer can create their own `local_overrides.yaml` without affecting others.

#### Nested Config Groups

Config groups can be nested in subdirectories:

```
configs/
└── model/
    ├── vision/
    │   ├── resnet18.yaml
    │   ├── resnet50.yaml
    │   └── vit.yaml
    └── nlp/
        ├── bert.yaml
        └── gpt.yaml
```

Usage with `/` separator:

```yaml
defaults:
  - model/vision: resnet50    # Loads configs/model/vision/resnet50.yaml
  - model/nlp: bert           # Loads configs/model/nlp/bert.yaml
```

From CLI:

```bash
python app.py model/vision=vit model/nlp=gpt
```

**Benefits:**
- Better organization for large projects
- Logical grouping of related configs
- Clearer config structure

#### Config Search Path

Hydra searches for configs in this order:

1. **Current working directory** (if config_path is relative)
2. **Specified config_path** in `@hydra.main()`
3. **Additional search paths** (can be added programmatically)

Example:

```python
@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: DictConfig):
    pass
```

This looks for configs in `configs/` relative to where the script is run.

### 3.8 Common Use Cases

#### Organizing Experiments

```
configs/
├── config.yaml
├── model/
│   ├── resnet18.yaml
│   ├── resnet50.yaml
│   └── vit.yaml
├── dataset/
│   ├── imagenet.yaml
│   ├── cifar10.yaml
│   └── mnist.yaml
├── optimizer/
│   ├── adam.yaml
│   ├── sgd.yaml
│   └── adamw.yaml
└── experiment/
    ├── baseline.yaml
    ├── ablation_1.yaml
    └── production.yaml
```

Then compose them:

```yaml
defaults:
  - model: resnet50
  - dataset: imagenet
  - optimizer: adam
  - experiment: baseline
  - _self_
```

#### Environment-Specific Configs

```
configs/
├── config.yaml
└── env/
    ├── local.yaml
    ├── dev.yaml
    └── production.yaml
```

Switch environments easily:

```bash
python app.py env=production
```

#### Modular Training Configurations

```yaml
#configs/config.yaml
defaults:
  - model: resnet50
  - data: imagenet
  - optimizer: adam
  - scheduler: cosine
  - augmentation: autoaugment
  - loss: cross_entropy
  - _self_

# Global settings
batch_size: 256
num_workers: 8
```

### 3.9 Best Practices and Tips

#### Best Practices

**1. Use meaningful group names:**
```
✓ Good: configs/model/, configs/optimizer/, configs/dataset/
✗ Bad: configs/m/, configs/opt/, configs/d/
```

**2. Keep configs focused:**
Each config file should configure one logical component
```yaml
# Good: configs/model/resnet50.yaml
name: resnet50
layers: 50
pretrained: true

# Bad: mixing concerns
name: resnet50
layers: 50
batch_size: 128  # This belongs in training config
learning_rate: 0.001  # This belongs in optimizer config
```

**3. Use `_self_` at the end:**
```yaml
defaults:
  - model: resnet50
  - _self_  # Your overrides apply last

# Your overrides
model:
  pretrained: false
```

**4. Document your defaults:**
```yaml
defaults:
  - model: resnet50          # Base model architecture
  - optimizer: adam          # Default optimizer
  - dataset: imagenet        # Training dataset
  - _self_
```

#### Common Pitfalls

**Pitfall 1: Forgetting the `+` when adding new groups**

```bash
# Wrong - errors if experiment not in defaults
python grouping_03/grouping.py experiment=experiment_with_resnet18

# Right - adds the group
python grouping_03/grouping.py +experiment=experiment_with_resnet18
```

**Pitfall 2: Not understanding `_self_` placement**

```yaml
# WRONG - _self_ at START means defaults override your values
defaults:
  - _self_                                # Apply primary config first
  - experiment: experiment_with_resnet50  # Then apply defaults (overrides!)

experiment:
  optimizer: SGD  # Gets overridden by experiment's adam → Result: adam ✗

# CORRECT - _self_ at END means your values win
defaults:
  - experiment: experiment_with_resnet50  # Apply defaults first
  - _self_                                # Then apply primary config

experiment:
  optimizer: SGD  # This applies last → Result: SGD ✓
```

**Remember:** Items later in the defaults list override earlier items. `_self_` represents when the primary config is applied.

**Pitfall 3: Circular dependencies**

```yaml
# configs/a.yaml
defaults:
  - b

# configs/b.yaml
defaults:
  - a

# This creates an infinite loop!
```

**Solution:** Design your config hierarchy carefully to avoid circular dependencies.

### 3.10 Test Commands

All examples in this section can be tested with the following commands. Run from the project root directory:

**Note:** All commands include `hydra.job.chdir=False` to prevent Hydra from creating output directories.

#### Adding config group from CLI
```bash
python grouping_03/grouping.py +experiment=experiment_with_resnet18 hydra.job.chdir=False
```

#### Using default config groups
```bash
python grouping_03/grouping.py --config-name=config_with_defaults hydra.job.chdir=False
```

#### Overriding defaults
```bash
python grouping_03/grouping.py --config-name=config_with_override hydra.job.chdir=False
```

#### Using _self_ for merge order
```bash
python grouping_03/grouping.py --config-name=config_with_self hydra.job.chdir=False
```

#### Composing multiple configs
```bash
python grouping_03/grouping.py --config-name=config_with_merge hydra.job.chdir=False
```

#### Switching configs from CLI
```bash
# Start with defaults, then switch experiment
python grouping_03/grouping.py --config-name=config_with_defaults experiment=experiment_with_resnet50 hydra.job.chdir=False
```

### 3.11 Files in grouping_03/ Directory

```
grouping_03/
├── configs/
│   ├── config.yaml                      # Empty base config
│   ├── config_with_defaults.yaml        # Config with default experiment
│   ├── config_with_override.yaml        # Config with override keyword
│   ├── config_with_self.yaml            # Config demonstrating _self_
│   ├── config_with_merge.yaml           # Config merging multiple files
│   ├── demo_config.yaml                 # Standalone config for merging
│   └── experiment/
│       ├── experiment_with_resnet18.yaml
│       └── experiment_with_resnet50.yaml
└── grouping.py                           # Main script
```

## 4. Multirun

Hydra's multirun feature allows you to run multiple experiments by sweeping over different configuration options using the `-m` flag. This is essential for hyperparameter tuning, model comparison, and ablation studies.

**Why use multirun?**
- **Hyperparameter sweeps**: Test multiple learning rates, batch sizes, or architectures automatically
- **Model comparisons**: Compare different models or loss functions systematically
- **Ablation studies**: Test the impact of different components
- **Reproducibility**: All experiment variations are tracked and logged
- **Automation**: Eliminate manual script modifications for each experiment

**How it works:**
Multirun creates a **cartesian product** of all parameter combinations. If you specify 2 experiments and 3 loss functions, you get 2 × 3 = 6 total jobs. Each job runs with a unique combination of parameters.

**All examples in this section are located in the `multirun_04/` directory.**

### 4.1 Basic Multirun Setup

We create configuration files for different experiments and loss functions:

```yaml
#multirun_04/configs/loss_function/softmax.yaml
name: softmax
```

```yaml
#multirun_04/configs/loss_function/cosface.yaml
name: cosface
margin: 0.5
```

```yaml
#multirun_04/configs/loss_function/arcface.yaml
name: arcface
margin: 0.8
```

```yaml
#multirun_04/configs/experiment/experiment_with_resnet18.yaml
model: resnet18
epochs: 100
batch_size: 128
lr: 0.001
optimizer: adam
scheduler: cosine
```

```yaml
#multirun_04/configs/experiment/experiment_with_resnet50.yaml
model: resnet50
epochs: 100
batch_size: 128
lr: 0.001
optimizer: adam
scheduler: cosine
```

```yaml
#multirun_04/configs/config.yaml
defaults:
  - experiment: experiment_with_resnet18
  - loss_function: arcface
  - _self_

experiment:
  optimizer: SGD

seed: 42
```

```python
#multirun_04/multirun.py
from omegaconf import OmegaConf, DictConfig
import hydra
from rich import print
import warnings
warnings.filterwarnings("ignore")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    main()
```

### 4.2 Running Multirun Experiments

Now we can run the experiments using the `-m` flag. Since we have 2 experiments and 3 loss functions, we will have 6 jobs in total.

**Understanding the Cartesian Product:**
- `experiment=experiment_with_resnet18,experiment_with_resnet50` → 2 options
- `loss_function=arcface,cosface,softmax` → 3 options
- Total combinations: 2 × 3 = **6 jobs**

Each job runs with a unique combination:
1. resnet18 + arcface
2. resnet18 + cosface
3. resnet18 + softmax
4. resnet50 + arcface
5. resnet50 + cosface
6. resnet50 + softmax

```bash
python multirun_04/multirun.py -m experiment=experiment_with_resnet18,experiment_with_resnet50 loss_function=arcface,cosface,softmax hydra.job.chdir=False
[2025-11-06 16:25:17,237][HYDRA] Launching 6 jobs locally
[2025-11-06 16:25:17,237][HYDRA]        #0 : experiment=experiment_with_resnet18 loss_function=arcface
experiment:
  model: resnet18
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: SGD
  scheduler: cosine
loss_function:
  name: arcface
  margin: 0.8
seed: 42

[2025-11-06 16:25:17,448][HYDRA]        #1 : experiment=experiment_with_resnet18 loss_function=cosface
experiment:
  model: resnet18
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: SGD
  scheduler: cosine
loss_function:
  name: cosface
  margin: 0.5
seed: 42

[2025-11-06 16:25:17,561][HYDRA]        #2 : experiment=experiment_with_resnet18 loss_function=softmax
experiment:
  model: resnet18
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: SGD
  scheduler: cosine
loss_function:
  name: softmax
seed: 42

[2025-11-06 16:25:17,680][HYDRA]        #3 : experiment=experiment_with_resnet50 loss_function=arcface
experiment:
  model: resnet50
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: SGD
  scheduler: cosine
loss_function:
  name: arcface
  margin: 0.8
seed: 42

[2025-11-06 16:25:17,830][HYDRA]        #4 : experiment=experiment_with_resnet50 loss_function=cosface
experiment:
  model: resnet50
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: SGD
  scheduler: cosine
loss_function:
  name: cosface
  margin: 0.5
seed: 42

[2025-11-06 16:25:17,939][HYDRA]        #5 : experiment=experiment_with_resnet50 loss_function=softmax
experiment:
  model: resnet50
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: SGD
  scheduler: cosine
loss_function:
  name: softmax
seed: 42
```

**Multirun Output Structure:**

When running multirun, Hydra creates a different directory structure than single runs:

```
multirun/
└── YYYY-MM-DD/
    └── HH-MM-SS/          # Timestamp of multirun launch
        ├── 0/             # Job #0 output
        │   └── .hydra/
        ├── 1/             # Job #1 output
        │   └── .hydra/
        ├── ...
        ├── 5/             # Job #5 output
        │   └── .hydra/
        └── multirun.yaml  # Overview of all jobs
```

Each job gets its own numbered directory, and the `multirun.yaml` file contains metadata about all runs.

### 4.3 Using Glob Syntax

To simplify the CLI command, we can use the `glob` syntax. This is especially useful when you have many config options or want pattern matching.

**Basic Glob Examples:**

```bash
# Match all experiments
experiment='glob(*)'

# Match specific patterns
experiment='glob(resnet*)'        # Matches resnet18, resnet50, etc.
loss_function='glob(*face)'       # Matches arcface, cosface

# Exclude patterns
loss_function='glob(*, exclude=soft*)'  # All except softmax
```

**Example: Excluding Softmax**

Notice that we use `exclude` to exclude the softmax loss function:

```bash
python multirun_04/multirun.py -m experiment='glob(*)' loss_function='glob(*, exclude=soft*)' hydra.job.chdir=False                                       
[2025-11-06 16:28:37,613][HYDRA] Launching 4 jobs locally
[2025-11-06 16:28:37,613][HYDRA]        #0 : experiment=experiment_with_resnet18 loss_function=arcface
experiment:
  model: resnet18
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: SGD
  scheduler: cosine
loss_function:
  name: arcface
  margin: 0.8
seed: 42

[2025-11-06 16:28:37,807][HYDRA]        #1 : experiment=experiment_with_resnet18 loss_function=cosface
experiment:
  model: resnet18
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: SGD
  scheduler: cosine
loss_function:
  name: cosface
  margin: 0.5
seed: 42

[2025-11-06 16:28:37,954][HYDRA]        #2 : experiment=experiment_with_resnet50 loss_function=arcface
experiment:
  model: resnet50
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: SGD
  scheduler: cosine
loss_function:
  name: arcface
  margin: 0.8
seed: 42

[2025-11-06 16:28:38,076][HYDRA]        #3 : experiment=experiment_with_resnet50 loss_function=cosface
experiment:
  model: resnet50
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: SGD
  scheduler: cosine
loss_function:
  name: cosface
  margin: 0.5
seed: 42
```

### 4.4 Advanced Sweep Syntax

#### Range Syntax

Hydra supports numerical ranges for sweeping over numeric hyperparameters:

```bash
# Sweep over learning rates
python multirun_04/multirun.py -m experiment.lr=0.001,0.0001,0.00001

# Using range() for batch sizes: range(start, stop, step)
python multirun_04/multirun.py -m experiment.batch_size=range(32,256,32)
# Generates: 32, 64, 96, 128, 160, 192, 224

# Combine with other parameters
python multirun_04/multirun.py -m experiment.lr=0.001,0.0001 experiment.batch_size=32,64,128
# Creates 2 × 3 = 6 jobs
```

#### Multiple Parameter Sweeps

You can sweep over any number of parameters:

```bash
# 3-dimensional sweep: 2 × 3 × 2 = 12 jobs
python multirun_04/multirun.py -m \
  experiment=experiment_with_resnet18,experiment_with_resnet50 \
  loss_function=arcface,cosface,softmax \
  experiment.lr=0.001,0.0001 \
  hydra.job.chdir=False
```

### 4.5 Common Use Cases

#### Hyperparameter Tuning

```bash
# Sweep over learning rate and weight decay
python multirun_04/multirun.py -m \
  experiment.lr=0.1,0.01,0.001,0.0001 \
  experiment.weight_decay=0.0,0.0001,0.001 \
  hydra.job.chdir=False
```

#### Model Architecture Comparison

```bash
# Compare different model architectures
python multirun_04/multirun.py -m \
  experiment='glob(*)' \
  hydra.job.chdir=False
```

#### Ablation Studies

Test the impact of different components:

```bash
# Test with/without different components
python multirun_04/multirun.py -m \
  loss_function='glob(*)' \
  experiment.dropout=0.0,0.1,0.2,0.3 \
  hydra.job.chdir=False
```

### 4.6 Best Practices and Tips

#### When to Use Multirun

**✓ Good use cases:**
- Hyperparameter sweeps (learning rate, batch size, etc.)
- Comparing different models or architectures
- Testing multiple random seeds for statistical significance
- Ablation studies to understand component importance

**✗ Avoid for:**
- Single experiments (use regular run instead)
- Very long-running jobs that would take too long in sequence
- When you need interactive debugging

#### Tips for Effective Multirun

**1. Start small, then expand:**
```bash
# First test with 2-3 combinations
python multirun_04/multirun.py -m experiment.lr=0.01,0.001

# Then expand after confirming it works
python multirun_04/multirun.py -m experiment.lr=0.1,0.01,0.001,0.0001
```

**2. Use glob for large sweeps:**
```bash
# Instead of listing all
python multirun_04/multirun.py -m experiment='glob(*)'
```

**3. Organize experiments logically:**
Create config groups for related experiments:
```
configs/
├── experiment/
│   ├── resnet_small/
│   ├── resnet_medium/
│   └── resnet_large/
```

**4. Track results systematically:**
- Each multirun creates a timestamped directory
- Use the job number to identify specific runs
- The `multirun.yaml` file contains all job configurations

#### Common Pitfalls

**Problem: Too many combinations**
```bash
# This creates 4 × 4 × 3 × 5 = 240 jobs!
python multirun_04/multirun.py -m \
  experiment.lr=0.1,0.01,0.001,0.0001 \
  experiment.batch_size=32,64,128,256 \
  loss_function='glob(*)' \
  experiment.dropout=0.0,0.1,0.2,0.3,0.4
```

**Solution:** Use sequential sweeps or reduce parameter space:
```bash
# First sweep over learning rate
python multirun_04/multirun.py -m experiment.lr=0.1,0.01,0.001,0.0001

# Then sweep over batch size with best LR
python multirun_04/multirun.py -m experiment.lr=0.001 experiment.batch_size=32,64,128,256
```

### 4.7 Test Commands

All examples in this section can be tested with the following commands. Run from the project root directory:

**Note:** All commands include `hydra.job.chdir=False` to prevent Hydra from creating output directories and changing the working directory.

#### Basic Multirun (6 jobs)
```bash
python multirun_04/multirun.py -m experiment=experiment_with_resnet18,experiment_with_resnet50 loss_function=arcface,cosface,softmax hydra.job.chdir=False
```

#### Using Glob Syntax (4 jobs - excludes softmax)
```bash
python multirun_04/multirun.py -m experiment='glob(*)' loss_function='glob(*, exclude=soft*)' hydra.job.chdir=False
```

#### Single Run (no multirun)
```bash
python multirun_04/multirun.py hydra.job.chdir=False
```

#### Range Sweep Example
```bash
# Sweep over learning rates using direct values
python multirun_04/multirun.py -m experiment.lr=0.1,0.01,0.001 hydra.job.chdir=False
```

### 4.8 Advanced Features

**Parallel Execution:**

By default, multirun jobs execute **sequentially** (one after another). For parallel execution, you can use Hydra launcher plugins:

- **Joblib Launcher**: Local parallel execution
- **SLURM Launcher**: For HPC clusters
- **Ray Launcher**: Distributed execution
- **AWS Batch Launcher**: Cloud execution

Example with Joblib (requires `hydra-joblib-launcher` plugin):
```bash
python multirun_04/multirun.py -m experiment='glob(*)' \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=4
```

**Note:** For the examples in this tutorial, we use sequential execution for simplicity.

### 4.9 Files in multirun_04/ Directory

```
multirun_04/
├── configs/
│   ├── config.yaml                        # Main config with defaults
│   ├── experiment/
│   │   ├── experiment_with_resnet18.yaml  # ResNet18 experiment config
│   │   └── experiment_with_resnet50.yaml  # ResNet50 experiment config
│   └── loss_function/
│       ├── arcface.yaml                   # ArcFace loss config
│       ├── cosface.yaml                   # CosFace loss config
│       └── softmax.yaml                   # Softmax loss config
└── multirun.py                            # Main script for multirun examples
```

## 5. Logging and Debugging

Hydra provides comprehensive logging and debugging tools to help you understand your configuration composition, track experiment outputs, and troubleshoot issues.

**Why use Hydra's logging and debugging?**
- **Config inspection**: View final merged configs before running your code
- **Composition debugging**: See exactly how configs are merged and where values come from
- **Experiment tracking**: Automatic logging of all runs with timestamps and saved configs
- **Troubleshooting**: Identify config conflicts, override issues, and composition problems
- **Reproducibility**: Every run is logged with its exact configuration and overrides

**How it works:**
Hydra automatically creates output directories for each run, saves your configuration, logs your application output, and provides CLI flags to inspect configs without running your code. This makes it easy to understand what's happening and reproduce results later.

**All examples in this section are located in the `logging_05/` directory.**

### 5.1 Output Directory Structure

When you run a Hydra application, it automatically creates an organized directory structure to store outputs from each run.

#### Single Run Directory Structure

By default, each run creates a timestamped directory:

```bash
outputs/
└── YYYY-MM-DD/           # Date of run
    └── HH-MM-SS/         # Time of run
        ├── .hydra/       # Hydra metadata (hidden directory)
        │   ├── config.yaml      # Final merged configuration
        │   ├── hydra.yaml       # Hydra's own configuration
        │   └── overrides.yaml   # List of CLI overrides used
        └── your_script.log      # Application logs (if logging enabled)
```

**Example:** Running the demo creates this structure:

```bash
python logging_05/logging_demo.py
```

This creates:
```bash
outputs/2025-11-14/21-08-57/
├── .hydra/
│   ├── config.yaml
│   ├── hydra.yaml
│   └── overrides.yaml
└── logging_demo.log
```

#### Multirun Directory Structure

When using multirun (`-m` flag), Hydra creates a different structure:

```bash
multirun/
└── YYYY-MM-DD/
    └── HH-MM-SS/         # Timestamp of multirun launch
        ├── 0/            # Job #0 output
        │   ├── .hydra/
        │   └── *.log
        ├── 1/            # Job #1 output
        │   ├── .hydra/
        │   └── *.log
        ├── 2/            # Job #2 output
        │   ├── .hydra/
        │   └── *.log
        └── multirun.yaml # Overview of all jobs
```

**Example:**
```bash
python logging_05/logging_demo.py -m training.batch_size=32,64,128
[2025-11-14 21:10:50,255][HYDRA] Launching 3 jobs locally
[2025-11-14 21:10:50,255][HYDRA] 	#0 : training.batch_size=32
[2025-11-14 21:10:50,437][HYDRA] 	#1 : training.batch_size=64
[2025-11-14 21:10:50,586][HYDRA] 	#2 : training.batch_size=128
```

Each job gets its own numbered directory with complete configuration and logs.

#### The .hydra Directory

The `.hydra/` directory contains three important files:

**1. config.yaml** - Your final merged configuration:
```bash
cat outputs/2025-11-14/21-08-57/.hydra/config.yaml
```
```yaml
training:
  batch_size: 128
  epochs: 30
  lr: 0.001
  optimizer: adam
model:
  name: resnet18
  pretrained: true
seed: 42
```

**2. overrides.yaml** - List of CLI overrides used:
```bash
# Run with overrides
python logging_05/logging_demo.py training.batch_size=256 model.name=resnet50

# Check the overrides file
cat outputs/2025-11-14/21-10-14/.hydra/overrides.yaml
```
```yaml
- training.batch_size=256
- model.name=resnet50
```

If no overrides were used, this file contains an empty list: `[]`

**3. hydra.yaml** - Hydra's internal configuration (output paths, logging settings, launcher config, etc.)

#### Controlling Output Directory Behavior

You can control where outputs are saved and whether Hydra changes the working directory:

```bash
# Prevent Hydra from changing working directory
python logging_05/logging_demo.py hydra.job.chdir=False

# Change output directory location
python logging_05/logging_demo.py hydra.run.dir=./my_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}

# Disable output directory creation entirely
python logging_05/logging_demo.py hydra.output_subdir=null hydra.run.dir=.
```

**When to use `hydra.job.chdir=False`:**
- When you need to access files relative to where you launched the script
- During development/debugging to keep outputs in one predictable location
- When integrating with tools that expect a specific working directory

### 5.2 Job Logging Configuration

Hydra provides flexible logging configuration through the `hydra/job_logging` config group.

#### Available Logging Options

You can choose from these logging modes:

- **`default`** - Logs to both console and file, standard formatting
- **`stdout`** - Logs to console only (stdout), no file created
- **`none`** - Minimal logging, console output only, no file
- **`disabled`** - Disables Hydra's logging configuration entirely (use your own)

#### Configuring Logging in Config Files

**Example 1: Enable logging with verbose mode**

```yaml
#logging_05/configs/config_with_logging.yaml
defaults:
  - _self_
  - override hydra/job_logging: default

training:
  batch_size: 128
  epochs: 30
  lr: 0.001
  optimizer: adam

model:
  name: resnet18
  pretrained: true

seed: 42

hydra:
  verbose: true  # Show detailed Hydra information
```

**Example 2: Disable file logging**

```yaml
#logging_05/configs/config_no_logging.yaml
defaults:
  - _self_
  - override hydra/job_logging: none

# ... rest of config

hydra:
  verbose: true  # Logs still appear in console, but no file is created
```

#### Python Logger Integration

Hydra automatically configures Python's logging module. You can use the standard `logging` module in your code:

```python
#logging_05/logging_demo.py
import logging
from omegaconf import DictConfig
import hydra

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    logger.info("INFO: Starting training...")
    logger.debug("DEBUG: Printing config...")
    logger.warning("WARNING: This is a warning message")

    logger.info(f"INFO: Training with batch_size={config.training.batch_size}")
    logger.info(f"INFO: Model: {config.model.name}")
    logger.info("INFO: Training completed!")
```

Running with logging enabled:

```bash
python logging_05/logging_demo.py --config-name=config_with_logging
[2025-11-14 21:08:57,482][__main__][INFO] - INFO: Starting training...
[2025-11-14 21:08:57,483][__main__][DEBUG] - DEBUG: Printing config...
[2025-11-14 21:08:57,483][__main__][WARNING] - WARNING: This is a warning message
[2025-11-14 21:08:57,548][__main__][INFO] - INFO: Training with batch_size=128
[2025-11-14 21:08:57,549][__main__][INFO] - INFO: Model: resnet18
[2025-11-14 21:08:57,549][__main__][INFO] - INFO: Training completed!
```

#### Log File Contents

The log file contains all logged messages:

```bash
cat outputs/2025-11-14/21-08-57/logging_demo.log
```
```
[2025-11-14 21:08:57,482][__main__][INFO] - INFO: Starting training...
[2025-11-14 21:08:57,483][__main__][DEBUG] - DEBUG: Printing config...
[2025-11-14 21:08:57,483][__main__][WARNING] - WARNING: This is a warning message
[2025-11-14 21:08:57,548][__main__][INFO] - INFO: Training with batch_size=128
[2025-11-14 21:08:57,549][__main__][INFO] - INFO: Model: resnet18
[2025-11-14 21:08:57,549][__main__][INFO] - INFO: Training completed!
```

**Note:** By default, the log level is `INFO`. `DEBUG` messages appear because we set `hydra.verbose: true`. Without verbose mode, `DEBUG` messages are filtered out.

#### Logging with No File Output

When using `job_logging: none`, logs still appear in console (if verbose is true) but no log file is created:

```bash
python logging_05/logging_demo.py --config-name=config_no_logging hydra.job.chdir=False
[2025-11-14 21:10:42,614][HYDRA] INFO: Starting training...
[2025-11-14 21:10:42,614][HYDRA] DEBUG: Printing config...
[2025-11-14 21:10:42,614][HYDRA] WARNING: This is a warning message
[2025-11-14 21:10:42,667][HYDRA] INFO: Training with batch_size=128
[2025-11-14 21:10:42,668][HYDRA] INFO: Model: resnet18
[2025-11-14 21:10:42,668][HYDRA] INFO: Training completed!
```

Notice the `[HYDRA]` prefix in the logs when using `none` mode.

### 5.3 Debugging Configs with --cfg

The `--cfg` flag lets you view your configuration without running your application code. This is essential for debugging configuration issues.

#### Available --cfg Options

- `--cfg job` - View your application's configuration only
- `--cfg hydra` - View Hydra's internal configuration
- `--cfg all` - View both job and Hydra configuration

#### Viewing Job Configuration

The most common debugging task is viewing the final merged configuration:

```bash
python logging_05/logging_demo.py --cfg job
```
```yaml
training:
  batch_size: 128
  epochs: 30
  lr: 0.001
  optimizer: adam
model:
  name: resnet18
  pretrained: true
seed: 42
```

**When your application doesn't run, use `--cfg job` to see what configuration Hydra assembled.**

#### Viewing Specific Packages with --package

Use `--package` to view only a specific section of the config:

```bash
python logging_05/logging_demo.py --cfg job --package training
```
```yaml
# @package training
batch_size: 128
epochs: 30
lr: 0.001
optimizer: adam
```

This is useful for large configs where you only want to see one component.

**Example with model package:**
```bash
python logging_05/logging_demo.py --cfg job --package model
```
```yaml
# @package model
name: resnet18
pretrained: true
```

#### Viewing Hydra's Configuration

See how Hydra itself is configured:

```bash
python logging_05/logging_demo.py --cfg hydra | head -30
```
```yaml
hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: ${hydra.job.num}
  launcher:
    _target_: hydra._internal.core_plugins.basic_launcher.BasicLauncher
  sweeper:
    _target_: hydra._internal.core_plugins.basic_sweeper.BasicSweeper
    max_batch_size: null
    params: null
  hydra_logging:
    version: 1
    formatters:
      simple:
        format: '[%(asctime)s][HYDRA] %(message)s'
    handlers:
      console:
        class: logging.StreamHandler
        formatter: simple
        stream: ext://sys.stdout
    root:
      level: INFO
      handlers:
      - console
    loggers:
      logging_example:
        level: DEBUG
    disable_existing_loggers: false
```

This shows:
- Output directory patterns (`outputs/${now:%Y-%m-%d}/...`)
- Launcher and sweeper configurations
- Logging configuration
- And much more

#### Viewing All Configuration

See both job and Hydra configs together:

```bash
python logging_05/logging_demo.py --cfg all | head -40
```

This outputs the complete configuration tree, starting with Hydra's config followed by your job config.

### 5.4 Resolving Interpolations with --resolve

Hydra supports variable interpolation in configs (e.g., `${model.name}`). Use `--resolve` to see the final resolved values.

#### Config with Interpolations

```yaml
#logging_05/configs/config_with_interpolation.yaml
training:
  batch_size: 128
  epochs: 30
  lr: 0.001
  optimizer: adam

model:
  name: resnet18
  pretrained: true
  input_size: 224

# Interpolations
experiment_name: ${model.name}_bs${training.batch_size}_lr${training.lr}
checkpoint_path: ./checkpoints/${experiment_name}
log_dir: ./logs/${experiment_name}

seed: 42
```

#### Without --resolve (shows interpolation syntax)

```bash
python logging_05/logging_demo.py --config-name=config_with_interpolation --cfg job
```
```yaml
training:
  batch_size: 128
  epochs: 30
  lr: 0.001
  optimizer: adam
model:
  name: resnet18
  pretrained: true
  input_size: 224
experiment_name: ${model.name}_bs${training.batch_size}_lr${training.lr}
checkpoint_path: ./checkpoints/${experiment_name}
log_dir: ./logs/${experiment_name}
seed: 42
```

#### With --resolve (shows final values)

```bash
python logging_05/logging_demo.py --config-name=config_with_interpolation --cfg job --resolve
```
```yaml
training:
  batch_size: 128
  epochs: 30
  lr: 0.001
  optimizer: adam
model:
  name: resnet18
  pretrained: true
  input_size: 224
experiment_name: resnet18_bs128_lr0.001
checkpoint_path: ./checkpoints/resnet18_bs128_lr0.001
log_dir: ./logs/resnet18_bs128_lr0.001
seed: 42
```

**Notice:** All `${...}` interpolations are resolved to their actual values.

**Use `--resolve` when:**
- Debugging interpolation errors
- Verifying that interpolated paths are correct
- Understanding what values will actually be used at runtime

### 5.5 Inspecting Config Composition with --info

The `--info` flag provides detailed information about how your configuration is composed. This is invaluable for debugging complex config hierarchies.

#### Available --info Options

- `--info defaults` - Show list of all default configs and where they come from
- `--info defaults-tree` - Show hierarchical tree of config composition
- `--info searchpath` - Show where Hydra searches for config files
- `--info config` - Show available config groups and options
- `--info plugins` - Show installed Hydra plugins
- `--info all` - Show all of the above

#### Viewing Defaults List

The defaults list shows every config file that's being loaded, in order:

```bash
python logging_05/logging_demo.py --config-name=config_with_defaults --info defaults
```
```
Defaults List
*************
| Config path                 | Package             | _self_ | Parent               |
--------------------------------------------------------------------------------------
| hydra/output/default        | hydra               | False  | hydra/config         |
| hydra/launcher/basic        | hydra.launcher      | False  | hydra/config         |
| hydra/sweeper/basic         | hydra.sweeper       | False  | hydra/config         |
| hydra/help/default          | hydra.help          | False  | hydra/config         |
| hydra/hydra_help/default    | hydra.hydra_help    | False  | hydra/config         |
| hydra/hydra_logging/default | hydra.hydra_logging | False  | hydra/config         |
| hydra/job_logging/default   | hydra.job_logging   | False  | hydra/config         |
| hydra/env/default           | hydra.env           | False  | hydra/config         |
| hydra/config                | hydra               | True   | <root>               |
| model/resnet18              | model               | False  | config_with_defaults |
| optimizer/adam              | optimizer           | False  | config_with_defaults |
| config_with_defaults        |                     | True   | <root>               |
--------------------------------------------------------------------------------------
```

**Understanding the columns:**
- **Config path**: Path to the config file
- **Package**: Where this config's content will be placed in the final config
- **_self_**: Whether this entry represents the current file's own content
- **Parent**: Which config file loaded this config

**Reading the table:**
- Hydra's internal configs are loaded first (hydra/...)
- Then your config groups are loaded (model/resnet18, optimizer/adam)
- The `_self_` entries show when each file's own content is merged

**Use this when:**
- You're not sure which config files are being loaded
- Config values aren't what you expected (check the order)
- Debugging package placement issues

#### Viewing Defaults Tree

The defaults tree shows the hierarchical structure of config composition:

```bash
python logging_05/logging_demo.py --config-name=config_with_defaults --info defaults-tree
```
```
Defaults Tree
*************
<root>:
  hydra/config:
    hydra/output: default
    hydra/launcher: basic
    hydra/sweeper: basic
    hydra/help: default
    hydra/hydra_help: default
    hydra/hydra_logging: default
    hydra/job_logging: default
    hydra/callbacks: null
    hydra/env: default
    _self_
  config_with_defaults:
    model: resnet18
    optimizer: adam
    _self_
```

**Understanding the tree:**
- `<root>` is the top level
- `hydra/config` loads all of Hydra's default configs
- `config_with_defaults` loads `model: resnet18` and `optimizer: adam`
- `_self_` shows where each config's own content is applied

**This is more intuitive than the defaults list for understanding:**
- Parent-child relationships between configs
- Which config loaded which sub-configs
- The overall composition structure

**Use this when:**
- Understanding complex config hierarchies
- Debugging composition order issues
- Learning how a new codebase structures its configs

#### Viewing Search Path

The search path shows where Hydra looks for config files:

```bash
python logging_05/logging_demo.py --info searchpath
```
```
Config search path
******************
| Provider       | Search path                                                            |
-------------------------------------------------------------------------------------------
| hydra          | pkg://hydra.conf                                                       |
| main           | file:///home/cyudhist/__projects__/Hydra-Essentials/logging_05/configs |
| hydra-colorlog | pkg://hydra_plugins.hydra_colorlog.conf                                |
| schema         | structured://                                                          |
-------------------------------------------------------------------------------------------
```

**Understanding the providers:**
- **hydra**: Hydra's built-in config files (launcher, sweeper, etc.)
- **main**: Your application's config directory (specified in `@hydra.main`)
- **hydra-colorlog**: Plugin providing colorized logging configs
- **schema**: For structured configs (Python dataclasses)

**Use this when:**
- Configs aren't being found
- Understanding which plugin configs are available
- Debugging config path issues

### 5.6 Hydra's Working Directory Behavior

By default, Hydra changes the working directory to the output directory. This can affect how your code accesses files.

#### Default Behavior

When you run a Hydra application without any flags:

```python
import os
print(f"Working directory: {os.getcwd()}")
```

```bash
python logging_05/logging_demo.py
```

The working directory will be something like:
```
/home/user/project/outputs/2025-11-14/21-08-57
```

**Why Hydra does this:**
- Makes it easy to save outputs (just write to current directory)
- All outputs from one run are in one place
- Prevents runs from overwriting each other's outputs

#### Disabling Directory Change

Prevent Hydra from changing the working directory:

```bash
python logging_05/logging_demo.py hydra.job.chdir=False
```

Now the working directory stays where you launched the script.

**When to use `hydra.job.chdir=False`:**
- During development/debugging
- When your code has hardcoded relative paths
- When you want outputs in a specific location
- When integrating with other tools

#### Accessing Original Working Directory

If Hydra changed the directory, you can still access the original location:

```python
from hydra.utils import get_original_cwd

original_dir = get_original_cwd()
data_path = f"{original_dir}/data/dataset.csv"
```

This is useful when you need to load data from the original location but want to save outputs to Hydra's output directory.

### 5.7 Verbose Mode and Hydra Runtime Information

The `hydra.verbose` setting controls how much information Hydra prints about its operation.

#### Enabling Verbose Mode

```yaml
hydra:
  verbose: true  # Show detailed Hydra information
```

Or from CLI:

```bash
python logging_05/logging_demo.py hydra.verbose=true
```

#### What Verbose Mode Shows

With `verbose: true`, Hydra prints:

1. **Hydra version and installed plugins**:
```
[2025-11-14 21:08:40,396][HYDRA] Hydra 1.3.2
[2025-11-14 21:08:40,396][HYDRA] ===========
[2025-11-14 21:08:40,396][HYDRA] Installed Hydra Plugins
[2025-11-14 21:08:40,396][HYDRA] ***********************
[2025-11-14 21:08:40,396][HYDRA] 	ConfigSource:
[2025-11-14 21:08:40,397][HYDRA] 	-------------
[2025-11-14 21:08:40,397][HYDRA] 		FileConfigSource
[2025-11-14 21:08:40,397][HYDRA] 		ImportlibResourcesConfigSource
[2025-11-14 21:08:40,397][HYDRA] 		StructuredConfigSource
```

2. **Config search path**:
```
[2025-11-14 21:08:40,399][HYDRA] Config search path
[2025-11-14 21:08:40,399][HYDRA] ******************
[2025-11-14 21:08:40,499][HYDRA] | Provider       | Search path                      |
[2025-11-14 21:08:40,499][HYDRA] | main           | file:///.../logging_05/configs   |
```

3. **Composition information**: Which configs are being loaded and merged

4. **Override information**: What overrides are being applied

**Use verbose mode when:**
- Debugging configuration issues
- Understanding what Hydra is doing
- Troubleshooting plugin or search path problems
- Learning how Hydra works

**Disable verbose mode when:**
- Running production code
- You want cleaner logs
- Running many experiments (less noise in logs)

### 5.8 Common Use Cases and Troubleshooting

Here are practical examples of using Hydra's debugging tools to solve real problems.

#### Use Case 1: "My config value is not what I expected"

**Problem:** You set `training.batch_size=256` but your code is using 128.

**Solution:** Check the final config and defaults order:

```bash
# 1. View final config
python logging_05/logging_demo.py training.batch_size=256 --cfg job

# 2. If value is still wrong, check the defaults tree
python logging_05/logging_demo.py training.batch_size=256 --info defaults-tree

# 3. Check if _self_ is in the wrong position
```

**Common causes:**
- Another config loaded later is overriding your value
- `_self_` is before the config that sets the value
- Typo in the override path

#### Use Case 2: "Hydra can't find my config file"

**Problem:** `Error: Could not find 'my_config' in config group 'model'`

**Solution:** Check the search path and available configs:

```bash
# 1. Verify search path is correct
python logging_05/logging_demo.py --info searchpath

# 2. Check if the file exists in the right location
ls -la logging_05/configs/model/

# 3. Verify the config group structure
```

**Common causes:**
- Config file is in the wrong directory
- Typo in filename
- `config_path` in `@hydra.main()` is incorrect

#### Use Case 3: "Interpolation isn't working"

**Problem:** `${model.name}` appears as literal text instead of resolving

**Solution:** Use `--resolve` to debug interpolations:

```bash
# 1. View interpolations without resolving
python logging_05/logging_demo.py --cfg job

# 2. Try to resolve them
python logging_05/logging_demo.py --cfg job --resolve

# If --resolve fails, it will show the error
```

**Common causes:**
- Reference to non-existent key
- Circular dependency in interpolations
- Wrong interpolation syntax (should be `${key.subkey}`, not `{key.subkey}`)

#### Use Case 4: "I don't know which config is being used"

**Problem:** Multiple config files, not sure which one is active

**Solution:** Check the defaults list:

```bash
python logging_05/logging_demo.py --info defaults
```

Look at the "Config path" column to see exactly which files are loaded.

#### Use Case 5: "My overrides aren't being applied"

**Problem:** CLI overrides seem to be ignored

**Solution:** Check the saved overrides and final config:

```bash
# 1. Run with overrides
python logging_05/logging_demo.py training.batch_size=512 model.name=resnet50

# 2. Check what overrides were saved
cat outputs/2025-11-14/*/.hydra/overrides.yaml

# 3. Check final config
cat outputs/2025-11-14/*/.hydra/config.yaml
```

**Common causes:**
- Typo in override path
- Config loaded after `_self_` is overriding your CLI override
- Wrong syntax (use `=` not `:`)

#### Use Case 6: "Debugging composition with packages"

**Problem:** Config content appears in the wrong place due to package directives

**Solution:** Use `--info defaults` to see package placement:

```bash
python logging_05/logging_demo.py --info defaults
```

Look at the "Package" column to see where each config's content will be placed.

### 5.9 Best Practices and Common Pitfalls

#### Best Practices

**1. Always use `--cfg job` before running experiments**

Before launching a large multirun, verify the config is correct:
```bash
python logging_05/logging_demo.py -m param1=a,b,c param2=1,2,3 --cfg job
```

**2. Use descriptive experiment names**

Override the output directory with meaningful names:
```bash
python logging_05/logging_demo.py \
  hydra.run.dir=outputs/resnet18_baseline_${now:%Y-%m-%d_%H-%M-%S}
```

**3. Keep verbose mode on during development**

```yaml
hydra:
  verbose: true  # Shows what Hydra is doing
```

Turn it off for production runs to reduce log noise.

**4. Use `--info defaults-tree` to understand complex configs**

When working with a new codebase, start with:
```bash
python app.py --info defaults-tree
```

This quickly shows you how configs are organized.

**5. Check saved configs after runs**

After important experiments, verify what config was actually used:
```bash
cat outputs/2025-11-14/21-10-14/.hydra/config.yaml
```

**6. Use `hydra.job.chdir=False` during development**

This keeps your working directory stable while you develop:
```bash
python logging_05/logging_demo.py hydra.job.chdir=False
```

#### Common Pitfalls

**Pitfall 1: Forgetting about the changed working directory**

**Problem:**
```python
# This fails because working directory changed
data = load_data("./data/dataset.csv")
```

**Solution:**
```python
from hydra.utils import get_original_cwd
data = load_data(f"{get_original_cwd()}/data/dataset.csv")
```

Or use `hydra.job.chdir=False`.

**Pitfall 2: Not checking interpolations before running**

**Problem:** Interpolations have errors, but you only discover this after a long training run.

**Solution:** Always use `--cfg job --resolve` first:
```bash
python app.py --cfg job --resolve
```

**Pitfall 3: Confusing job config with Hydra config**

**Problem:** Trying to override Hydra's settings like job config:

```bash
# Wrong
python app.py run.dir=./my_outputs

# Right
python app.py hydra.run.dir=./my_outputs
```

**Remember:** Hydra's settings are under the `hydra` key.

**Pitfall 4: Not understanding the defaults list order**

**Problem:** Config values are wrong because you don't understand merge order.

**Solution:** Use `--info defaults` to see the exact order:
```bash
python app.py --info defaults
```

Later entries override earlier ones.

**Pitfall 5: Losing track of experiments**

**Problem:** Hundreds of timestamped directories, can't find the right run.

**Solution:**
- Use meaningful output directories
- Add experiment name to config
- Use git commit hash in output path:

```yaml
hydra:
  run:
    dir: outputs/${experiment_name}/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

**Pitfall 6: Not using version control for configs**

**Problem:** Config files change over time, can't reproduce old results.

**Solution:**
- Keep configs in git
- The `.hydra/config.yaml` file saves the exact config used
- Add git hash to experiment name for tracking

### 5.10 Test Commands

All examples in this section can be tested with the following commands. Run from the project root directory.

**Note:** All commands include `hydra.job.chdir=False` where appropriate to prevent Hydra from creating output directories and changing the working directory during testing.

#### Basic Debugging

```bash
# View job config
python logging_05/logging_demo.py --cfg job

# View specific package
python logging_05/logging_demo.py --cfg job --package training

# View Hydra's configuration
python logging_05/logging_demo.py --cfg hydra | head -50

# View all configuration
python logging_05/logging_demo.py --cfg all | head -50
```

#### Config Composition Debugging

```bash
# View defaults list
python logging_05/logging_demo.py --config-name=config_with_defaults --info defaults

# View defaults tree
python logging_05/logging_demo.py --config-name=config_with_defaults --info defaults-tree

# View search path
python logging_05/logging_demo.py --info searchpath
```

#### Interpolation Debugging

```bash
# View interpolations (not resolved)
python logging_05/logging_demo.py --config-name=config_with_interpolation --cfg job

# View interpolations (resolved)
python logging_05/logging_demo.py --config-name=config_with_interpolation --cfg job --resolve
```

#### Running with Different Logging Modes

```bash
# Run with default logging (creates log file and output directory)
python logging_05/logging_demo.py --config-name=config_with_logging

# Run with no logging (console only, verbose mode, no output directory)
python logging_05/logging_demo.py --config-name=config_no_logging hydra.job.chdir=False

# Run with overrides
python logging_05/logging_demo.py training.batch_size=256 model.name=resnet50

# Run multirun
python logging_05/logging_demo.py -m training.batch_size=32,64,128 hydra.job.chdir=False
```

#### Checking Output Files

```bash
# List recent output directories
ls -lt outputs/$(date +%Y-%m-%d)/ 2>/dev/null || echo "No outputs today"

# View log file from latest run
find outputs -name "*.log" -type f | tail -1 | xargs cat

# View saved config from latest run
find outputs -name "config.yaml" -path "*/.hydra/*" -type f | tail -1 | xargs cat

# View overrides from latest run
find outputs -name "overrides.yaml" -path "*/.hydra/*" -type f | tail -1 | xargs cat
```

### 5.11 Files in logging_05/ Directory

```
logging_05/
├── configs/
│   ├── config.yaml                      # Basic config
│   ├── config_with_logging.yaml         # Config with logging enabled and verbose mode
│   ├── config_no_logging.yaml           # Config with logging disabled (console only)
│   ├── config_with_interpolation.yaml   # Config with interpolations for --resolve demo
│   ├── config_with_defaults.yaml        # Config with defaults for --info demos
│   ├── model/
│   │   └── resnet18.yaml                # Model config group
│   └── optimizer/
│       ├── adam.yaml                    # Adam optimizer config
│       └── sgd.yaml                     # SGD optimizer config
└── logging_demo.py                      # Main demo script with logging examples
```

## 6. Instantiate

Hydra's `instantiate` function allows you to create Python objects directly from configuration files. This is a powerful feature that enables you to configure complex object hierarchies declaratively.

**All examples in this section are located in the `instantiate_06/` directory.** See `instantiate_06/README.md` for quick reference commands.

### 6.1 Basic Instantiation

We use the `_target_` key to specify the class to instantiate, and other keys become constructor arguments.

```yaml
#instantiate_06/instantiate_config.yaml
my_class:
  # we are specifying the class to instantiate
  _target_: instantiate_06.instantiate.MyClass
  # we are passing the name to the class
  name: Paul
```

```python
#instantiate_06/instantiate.py
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate


class MyClass:
    def __init__(self, name: str):
        self.name = name

    def say_hello(self):
        print(f"Hello, {self.name}!")


@hydra.main(config_path=".", config_name="instantiate_config", version_base=None)
def main(config: DictConfig):
    my_class = MyClass(name="John")
    my_class.say_hello()

    # instantiate the class using hydra
    my_class = instantiate(config.my_class)
    my_class.say_hello()


if __name__ == "__main__":
    main()
```

Running the script as a module with `-m` flag will print:

```bash
python -m instantiate_06.instantiate
Hello, John!
Hello, Paul!
```

### 6.2 Partial Instantiation

The `_partial_` key creates a partial function instead of immediately instantiating the object. This is useful for classes that require parameters you don't have yet (like optimizers needing model parameters).

```yaml
#instantiate_06/instantiate_config.yaml
optimizer:
  # we are specifying the class to instantiate
  _target_: torch.optim.Adam
  # we are telling hydra that this is a partial config
  # so it will not raise an error if the config is not complete
  _partial_: true
  # we are passing the learning rate to the class
  lr: 0.01
  betas: [0.1, 0.9]
  eps: 1e-6
```

The partial optimizer can be called later with the missing parameters:

```python
#instantiate_06/instantiate.py
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
import warnings
warnings.filterwarnings("ignore")

from rich import print


class MyClass:
    def __init__(self, name: str):
        self.name = name

    def say_hello(self):
        print(f"Hello, {self.name}!")


@hydra.main(config_path=".", config_name="instantiate_config", version_base=None)
def main(config: DictConfig):
    my_class = MyClass(name="John")
    my_class.say_hello()

    # instantiate the class using hydra
    my_class = instantiate(config.my_class)
    my_class.say_hello()

    # instantiate the optimizer using hydra
    print("Instantiating optimizer...")
    parameters = torch.nn.Parameter(torch.randn(10))
    print("Parameters:", parameters)
    partial_optimizer = instantiate(config.optimizer)
    print("Partial optimizer:", partial_optimizer)
    optimizer = partial_optimizer([parameters])
    print(optimizer)


if __name__ == "__main__":
    main()
```

Running the script will print:

```bash
python -m instantiate_06.instantiate
Instantiating optimizer...
Parameters: Parameter containing:
tensor([-0.6364,  1.1314,  0.6071,  0.0829,  0.6568,  0.7903, -0.5078,  1.7053,
        -0.0383,  1.4658], requires_grad=True)
Partial optimizer: functools.partial(<class 'torch.optim.adam.Adam'>, lr=0.01, betas=[0.1, 0.9], eps=1e-06)
Adam (
Parameter Group 0
    amsgrad: False
    betas: [0.1, 0.9]
    capturable: False
    decoupled_weight_decay: False
    differentiable: False
    eps: 1e-06
    foreach: None
    fused: None
    lr: 0.01
    maximize: False
    weight_decay: 0
)
```

### 6.3 Recursive Instantiation

By default, Hydra recursively instantiates nested configs that contain `_target_`. This allows you to build complex object hierarchies.

```yaml
#instantiate_06/instantiate_recursive_config.yaml
model:
  _target_: instantiate_06.instantiate_recursive.SimpleModel
  # These nested configs will also be instantiated
  backbone:
    _target_: instantiate_06.instantiate_recursive.Backbone
    hidden_size: 512
  head:
    _target_: instantiate_06.instantiate_recursive.Head
    num_classes: 10
```

```python
#instantiate_06/instantiate_recursive.py
class Backbone:
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        print(f"Backbone created with hidden_size={hidden_size}")


class Head:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        print(f"Head created with num_classes={num_classes}")


class SimpleModel:
    def __init__(self, backbone, head):
        self.backbone = backbone
        self.head = head
        print("SimpleModel created with backbone and head")


@hydra.main(config_path=".", config_name="instantiate_recursive_config", version_base=None)
def main(config: DictConfig):
    # This will recursively instantiate backbone and head, then pass them to SimpleModel
    model = instantiate(config.model)
    print(f"Model backbone hidden_size: {model.backbone.hidden_size}")
    print(f"Model head num_classes: {model.head.num_classes}")
```

Output:

```bash
python -m instantiate_06.instantiate_recursive
Backbone created with hidden_size=512
Head created with num_classes=10
SimpleModel created with backbone and head
Model backbone hidden_size: 512
Model head num_classes: 10
```

### 6.4 Advanced Parameters

#### Using `_recursive_` to Control Nested Instantiation

You can disable recursive instantiation with `_recursive_: false`:

```yaml
#instantiate_06/instantiate_no_recursive_config.yaml
model:
  _target_: instantiate_06.instantiate_no_recursive.ModelWithConfig
  _recursive_: false  # Don't instantiate nested configs
  # This will be passed as a DictConfig, not an instantiated object
  layer_config:
    _target_: torch.nn.Linear
    in_features: 10
    out_features: 5
```

```python
#instantiate_06/instantiate_no_recursive.py
class ModelWithConfig:
    def __init__(self, layer_config):
        # layer_config is a DictConfig, not an instantiated Linear layer
        print(f"Received config: {layer_config}")
        # You can manually instantiate it later if needed
        from hydra.utils import instantiate
        self.layer = instantiate(layer_config)
```

Run it:

```bash
python -m instantiate_06.instantiate_no_recursive
```

#### Using `_args_` for Positional Arguments

```yaml
#instantiate_06/instantiate_args_config.yaml
my_function:
  _target_: instantiate_06.instantiate_args.my_function
  # Positional arguments
  _args_:
    - "first positional arg"
    - "second positional arg"
  # Keyword arguments
  keyword_arg: "keyword value"
```

```python
#instantiate_06/instantiate_args.py
def my_function(pos1, pos2, keyword_arg=None):
    print(f"pos1={pos1}, pos2={pos2}, keyword_arg={keyword_arg}")


@hydra.main(config_path=".", config_name="instantiate_args_config", version_base=None)
def main(config: DictConfig):
    instantiate(config.my_function)
```

Output:

```bash
python -m instantiate_06.instantiate_args
pos1=first positional arg, pos2=second positional arg, keyword_arg=keyword value
```

#### Using `_convert_` to Control Container Conversion

The `_convert_` parameter controls how OmegaConf containers are converted:

```yaml
#instantiate_06/instantiate_convert_config.yaml
my_class:
  _target_: instantiate_06.instantiate_convert.MyClass
  _convert_: all  # Options: none, partial, all (default)
  data:
    nested: value
```

- `_convert_: "none"` - Pass OmegaConf containers as-is
- `_convert_: "partial"` - Convert only the top-level container
- `_convert_: "all"` - Convert all nested containers to dict/list (default)

Run it:

```bash
python -m instantiate_06.instantiate_convert
```

### 6.5 Additional Utilities

#### Using `get_class()` to Get Class Without Instantiation

Sometimes you want to get the class itself without instantiating it:

```python
#instantiate_06/instantiate_get_class.py
from hydra.utils import get_class

@hydra.main(config_path=".", config_name="instantiate_config", version_base=None)
def main(config: DictConfig):
    # Get the class without instantiating
    optimizer_class = get_class("torch.optim.Adam")
    print(f"Got class: {optimizer_class}")

    # Use it later with your own parameters
    model = torch.nn.Linear(10, 5)
    optimizer = optimizer_class(model.parameters(), lr=0.001)
    print(f"Created optimizer: {optimizer}")
```

Output:

```bash
python -m instantiate_06.instantiate_get_class
Got class: <class 'torch.optim.adam.Adam'>
Created optimizer: Adam (...)
```

#### Using `call()` to Call Functions

The `call()` function is similar to `instantiate()` but for calling functions instead of constructing objects:

```python
#instantiate_06/instantiate.py
from hydra.utils import call

@hydra.main(config_path=".", config_name="instantiate_config", version_base=None)
def main(config: DictConfig):
    # Call a function from config
    result = call(config.my_function)
```

### 6.6 Common Pitfalls and Best Practices

#### Pitfall 1: Forgetting `_partial_: true` for Optimizers

**Problem:**

```yaml
optimizer:
  _target_: torch.optim.Adam
  lr: 0.001
```

This will fail because `Adam` requires `params` as the first argument, which you don't have yet.

**Solution:**

```yaml
optimizer:
  _target_: torch.optim.Adam
  _partial_: true  # Create a partial function
  lr: 0.001
```

#### Pitfall 2: Incorrect Import Path in `_target_`

**Problem:**

```yaml
model:
  _target_: MyModel  # Wrong - no module path
```

**Solution:**

```yaml
model:
  _target_: models.MyModel  # Correct - full import path
```

The path must be importable from your Python environment. Use the same path you would use in `from X import Y`.

#### Pitfall 3: Mixing Positional and Keyword Arguments Incorrectly

**Problem:**

```yaml
my_class:
  _target_: MyClass
  _args_:
    - arg1
  arg1: different_value  # Conflict!
```

**Solution:** Use either positional arguments (`_args_`) OR keyword arguments, not both for the same parameter.

#### Pitfall 4: Not Understanding When Recursive Instantiation Happens

By default, any nested dict with `_target_` gets instantiated. If you want to pass a config dict without instantiating it, use `_recursive_: false`.

**Example:**

```yaml
model:
  _target_: MyModel
  _recursive_: false
  config:
    _target_: SomeClass  # This won't be instantiated
```

#### Best Practice: Use Type Hints in Your Classes

Type hints make it easier to understand what parameters your classes expect:

```python
class MyModel:
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        self.hidden_size = hidden_size
        self.dropout = dropout
```

This helps when writing configs, as you know what types to provide.

### 6.7 Test Commands

All examples in this section can be tested with the following commands. Run from the project root directory:

**Note:** All commands include `hydra.job.chdir=False` to prevent Hydra from creating output directories and changing the working directory. This makes the examples easier to run and test.

#### 6.1 Basic Instantiation
```bash
python -m instantiate_06.instantiate hydra.job.chdir=False
```

#### 6.2 Partial Instantiation
```bash
python -m instantiate_06.instantiate hydra.job.chdir=False
```
(Same script as 6.1, demonstrates both basic and partial instantiation)

#### 6.3 Recursive Instantiation
```bash
python -m instantiate_06.instantiate_recursive hydra.job.chdir=False
```

#### 6.4 Advanced Parameters

**Using `_recursive_` to control nested instantiation:**
```bash
python -m instantiate_06.instantiate_no_recursive hydra.job.chdir=False
```

**Using `_args_` for positional arguments:**
```bash
python -m instantiate_06.instantiate_args hydra.job.chdir=False
```

**Using `_convert_` to control container conversion:**
```bash
python -m instantiate_06.instantiate_convert hydra.job.chdir=False
```

#### 6.5 Additional Utilities

**Using `get_class()` to get class without instantiation:**
```bash
python -m instantiate_06.instantiate_get_class hydra.job.chdir=False
```

### 6.8 Files in instantiate_06/ Directory

- `instantiate.py` + `instantiate_config.yaml` - Basic and partial instantiation
- `instantiate_recursive.py` + `instantiate_recursive_config.yaml` - Recursive instantiation
- `instantiate_no_recursive.py` + `instantiate_no_recursive_config.yaml` - Disable recursive instantiation
- `instantiate_args.py` + `instantiate_args_config.yaml` - Positional arguments with `_args_`
- `instantiate_convert.py` + `instantiate_convert_config.yaml` - Container conversion with `_convert_`
- `instantiate_get_class.py` - Get class without instantiation using `get_class()`

------------------------------------------------------------------------------------------------------------------------------------------

## 7. Packages

In Hydra, a "package" refers to where a config group's content is placed in the final configuration structure. By default, a config from the `task` group is placed under the `task` key in the final config. The package directive (`@`) allows you to control this placement.

### 7.1 Project Structure

We start with a project structure as such:

```bash
├── configs
│   ├── config.yaml
│   └── task
│       ├── mnist_classification.yaml
│       └── model
│           ├── adapter
│           │   └── mnist_classification_resnet18.yaml
│           ├── backbone
│           │   └── resnet18.yaml
│           ├── head
│           │   └── identity_head.yaml
│           └── simple_model.yaml
└── packages.py
```

Where packages.py is our main script that prints the config.

Our config.yaml file is as such:
```yaml
#configs/config.yaml
defaults:
  - task: mnist_classification
```

Our mnist_classification.yaml file is as such:
```yaml
#configs/task/mnist_classification.yaml
defaults:
  - model: simple_model
```

Our model/adapter/mnist_classification_resnet18.yaml file is as such:

```yaml
#configs/task/model/adapter/mnist_classification_resnet18.yaml
_target_: some_file.LinearAdapter
in_features: 512
out_features: 10
```

Our model/backbone/resnet18.yaml file is as such:
```yaml
#configs/task/model/backbone/resnet18.yaml
_target_: some_file.ResNet18
```

Our model/head/identity_head.yaml file is as such:
```yaml
#configs/task/model/head/identity_head.yaml
_target_: some_file.IdentityHead
```

Our simple_model.yaml file is as such:
```yaml
#configs/task/model/simple_model.yaml
defaults:
  - backbone: resnet18
  - adapter: mnist_classification_resnet18
  - head: identity_head

constant: 9
```

When we run the packages.py file, it will print the config. Notice how the content from the `task` group is placed under the `task` key:

```bash
task:
  model:
    backbone:
      _target_: some_file.ResNet18
    adapter:
      _target_: some_file.LinearAdapter
      in_features: 512
      out_features: 10
    head:
      _target_: some_file.IdentityHead
    constant: 9
```

### 7.2 Using @ Syntax to Change Package Placement

We can change where the config content is placed using the `@` syntax. The format is: `group@package_name: config_name`

This loads the same `mnist_classification` config from the `task` group three times, but places each instance under a different key in the final config:

```yaml
#configs/config.yaml
defaults:
  - task: mnist_classification              # placed at 'task' key (default)
  - task@second_task: mnist_classification  # placed at 'second_task' key
  - task@third_task: mnist_classification   # placed at 'third_task' key
```

This will print:

```bash
task:
  model:
    backbone:
      _target_: some_file.ResNet18
    adapter:
      _target_: some_file.LinearAdapter
      in_features: 512
      out_features: 10
    head:
      _target_: some_file.IdentityHead
    constant: 9
second_task:
  model:
    backbone:
      _target_: some_file.ResNet18
    adapter:
      _target_: some_file.LinearAdapter
      in_features: 512
      out_features: 10
    head:
      _target_: some_file.IdentityHead
    constant: 9
third_task:
  model:
    backbone:
      _target_: some_file.ResNet18
    adapter:
      _target_: some_file.LinearAdapter
      in_features: 512
      out_features: 10
    head:
      _target_: some_file.IdentityHead
    constant: 9
```

### 7.3 Loading Nested Configs with Absolute Paths

You can also load configs from nested directories using absolute paths (starting with `/`). The format is: `/absolute/path/to/config@package_name`

Here we load a specific model config that's nested deep in the folder structure and place it at the root level under `my_simple_model`:

```yaml
#configs/config.yaml
defaults:
  - task: mnist_classification

  # /task/model/simple_model = absolute path from configs/ root
  # @my_simple_model = place content at 'my_simple_model' key instead of nested under task.model
  - /task/model/simple_model@my_simple_model
```

When we run the packages.py file, it will print the config. Notice how `my_simple_model` appears at the root level, not nested:
```bash
task:
  model:
    backbone:
      _target_: some_file.ResNet18
    adapter:
      _target_: some_file.LinearAdapter
      in_features: 512
      out_features: 10
    head:
      _target_: some_file.IdentityHead
    constant: 9
my_simple_model:
  backbone:
    _target_: some_file.ResNet18
  adapter:
    _target_: some_file.LinearAdapter
    in_features: 512
    out_features: 10
  head:
    _target_: some_file.IdentityHead
  constant: 9
```

### 7.4 Using the `# @package` Directive Inside Config Files

The `# @package` directive is placed **inside** a config file to control where that file's content will be placed in the final configuration. This is useful when you want a config file to always be placed at a specific location, regardless of how it's loaded.

```yaml
#configs/task/model/backbone/resnet18.yaml
# @package foo.bar

# The backbone content will no longer appear under task.model.backbone
# Instead, it will be placed at foo.bar in the final config
_target_: some_file.ResNet18
```

When we run the packages.py file:

```bash
foo:
  bar:
    _target_: some_file.ResNet18
task:
  model:
    adapter:
      _target_: some_file.LinearAdapter
      in_features: 512
      out_features: 10
    head:
      _target_: some_file.IdentityHead
    constant: 9
```

Common `# @package` directives:
- `# @package _global_` - Place content at the root level
- `# @package foo.bar` - Place content at the `foo.bar` path
- `# @package _group_` - Use the default package for this group

### 7.5 Using `# @package _global_` to Avoid Nesting

A common use case for packages is to avoid deep nesting by placing config content directly at the root level using `# @package _global_`. This is especially useful for global settings like seed, debug flags, or logging configurations.

Let's create a new config file for global settings:

```yaml
#configs/task/global_settings.yaml
# @package _global_

# These fields will appear at the root level, not under task.global_settings
seed: 42
debug: true
verbose: false
```

Update the main config to include it:

```yaml
#configs/config.yaml
defaults:
  - task: mnist_classification
  - task/global_settings  # Note: we use task/global_settings to reference the file
```

When we run packages.py, the output will show `seed`, `debug`, and `verbose` at the root level:

```bash
seed: 42
debug: true
verbose: false
task:
  model:
    backbone:
      _target_: some_file.ResNet18
    adapter:
      _target_: some_file.LinearAdapter
      in_features: 512
      out_features: 10
    head:
      _target_: some_file.IdentityHead
    constant: 9
```

Without `# @package _global_`, the output would have been:

```bash
task:
  global_settings:
    seed: 42
    debug: true
    verbose: false
  model:
    backbone:
      _target_: some_file.ResNet18
    adapter:
      _target_: some_file.LinearAdapter
      in_features: 512
      out_features: 10
    head:
      _target_: some_file.IdentityHead
    constant: 9
```

### 7.6 Why and When to Use Packages

Packages are a powerful feature in Hydra that help you organize and structure your configurations. Here are the main use cases:

#### Use Case 1: Avoiding Deep Nesting

Without packages, configs can become deeply nested and hard to access:

```python
# Hard to access deeply nested values
model_name = config.experiments.task.model.backbone.name
```

With `# @package _global_`, you can flatten the structure:

```python
# Easier to access
model_name = config.model_name
```

#### Use Case 2: Reusing Configs in Different Locations

You might want to use the same configuration in multiple places within your config. For example, using the same model config for both training and validation:

```yaml
#configs/config.yaml
defaults:
  - task@training.model: resnet18
  - task@validation.model: resnet18
```

Result:
```yaml
training:
  model:
    # resnet18 config here
validation:
  model:
    # same resnet18 config here
```

#### Use Case 3: Building Modular, Composable Configurations

Packages allow you to build configs from independent, reusable components that can be combined in different ways:

```yaml
#configs/config.yaml
defaults:
  - dataset@data.train: imagenet
  - dataset@data.val: imagenet
  - dataset@data.test: coco
  - model@training: resnet50
  - optimizer@training: adam
```

This creates a well-organized structure where each component lives in its logical place.

#### Use Case 4: Organizing Related Configs Under Custom Namespaces

Group related configurations together for better organization:

```yaml
#configs/config.yaml
defaults:
  - optimizer@hyperparameters.optim: adam
  - scheduler@hyperparameters.sched: cosine
  - augmentation@hyperparameters.aug: standard
```

Result:
```yaml
hyperparameters:
  optim:
    # adam optimizer config
  sched:
    # cosine scheduler config
  aug:
    # augmentation config
```

#### When to Use Packages:

- **Use `@` in defaults** when you want to control where a config group is placed at load time
- **Use `# @package` in files** when you want a config to always be placed at a specific location, regardless of how it's loaded
- **Use `# @package _global_`** for truly global settings (seed, debug flags, logging levels) that should be at root level
- **Use dot notation (`@parent.child`)** when building hierarchical structures with multiple levels

### 7.7 Package Conflicts and Merge Behavior

When multiple configs target the same package location, Hydra merges them. The **last** config in the defaults list wins for conflicting keys (later configs override earlier ones).

#### Example: Multiple Configs to Same Location

Let's create two config files that will both target `_global_`:

```yaml
#configs/task/settings_a.yaml
# @package _global_

seed: 42
learning_rate: 0.001
batch_size: 128
```

```yaml
#configs/task/settings_b.yaml
# @package _global_

seed: 99
batch_size: 256
epochs: 100
```

Now load both in the main config:

```yaml
#configs/config.yaml
defaults:
  - task: mnist_classification
  - task/settings_a
  - task/settings_b  # This is loaded AFTER settings_a
```

When we run packages.py:

```bash
seed: 99             # settings_b overrides settings_a
learning_rate: 0.001 # from settings_a (no conflict)
batch_size: 256      # settings_b overrides settings_a
epochs: 100          # from settings_b (no conflict)
task:
  model:
    # ... task config here
```

**Key Points:**
- `seed` was in both configs - `settings_b` (99) wins because it's loaded last
- `batch_size` was in both configs - `settings_b` (256) wins
- `learning_rate` only in `settings_a` - kept
- `epochs` only in `settings_b` - kept
- Configs are **merged**, not replaced entirely

#### Controlling Merge Order with Defaults

The order in the defaults list matters. You can also use `_self_` to control when the current file's own content is merged.

**Note:** `_self_` is a special keyword that refers to the content in the **current config file** (the file where `_self_` appears).

Here's a complete example:

```yaml
#configs/config.yaml
defaults:
  - task: mnist_classification
  - task/settings_a  # Loaded first: seed=42, learning_rate=0.001, batch_size=128
  - task/settings_b  # Loaded second: seed=99, batch_size=256, epochs=100
  - _self_           # Apply THIS file's content last (below)

# The content below is what _self_ refers to:
seed: 1000
batch_size: 512
custom_field: hello
```

When we run packages.py:

```bash
seed: 1000           # _self_ wins (loaded last)
batch_size: 512      # _self_ wins (loaded last)
learning_rate: 0.001 # from settings_a (no conflict)
epochs: 100          # from settings_b (no conflict)
custom_field: hello  # from _self_ (no conflict)
task:
  model:
    # ... task config here
```

**By default, `_self_` is implicitly at the end.** You only need to specify it explicitly when you want to change its position:

```yaml
#configs/config.yaml
defaults:
  - _self_           # Apply this file's content FIRST
  - task/settings_a  # Then settings_a (will override this file)
  - task/settings_b  # Then settings_b (will override everything above)

# With _self_ first, settings_b would win conflicts
seed: 1000
```

In this case, the final `seed` would be 99 (from settings_b) instead of 1000.

#### Avoiding Conflicts

To avoid unintentional conflicts:

1. **Use specific package locations** instead of `_global_`:
```yaml
# @package config.training
seed: 42

# @package config.model
seed: 99
```

2. **Use different namespaces** with `@` syntax:
```yaml
defaults:
  - task/settings_a@training
  - task/settings_b@validation
```

### 7.8 Precedence: `@` in Defaults vs `# @package` in Files

What happens when you use both `@` syntax in the defaults list AND `# @package` directive inside the config file? Which one wins?

**Answer: The `# @package` directive inside the file takes precedence.**

#### Example: Conflicting Package Directives

Let's create a config file with a `# @package` directive:

```yaml
#configs/task/my_config.yaml
# @package _global_

# This file says: "I want to be placed at _global_"
my_setting: 100
another_setting: 200
```

Now try to load it with a different package location using `@` in defaults:

```yaml
#configs/config.yaml
defaults:
  - task: mnist_classification
  - task/my_config@custom_location  # Trying to place it at 'custom_location'
```

When we run packages.py:

```bash
# The # @package _global_ directive WINS
my_setting: 100
another_setting: 200
task:
  model:
    # ... task config here
```

**Notice:** The content appears at the root level (`_global_`), NOT at `custom_location`. The `# @package` directive inside the file overrides the `@` syntax in defaults.

#### Why Does `# @package` Win?

The `# @package` directive is more explicit and specific - it's a property of the config file itself. The `@` syntax in defaults is just a suggestion of where to place the content, but the file's own `# @package` directive has final say.

#### What If There's No `# @package` Directive?

If the config file doesn't have a `# @package` directive, then the `@` syntax in defaults determines the placement:

```yaml
#configs/task/settings.yaml
# No @package directive

setting_x: 50
setting_y: 75
```

```yaml
#configs/config.yaml
defaults:
  - task: mnist_classification
  - task/settings@custom_location  # This will work since no @package in the file
```

Output:

```bash
custom_location:
  setting_x: 50
  setting_y: 75
task:
  model:
    # ... task config here
```

#### Practical Tip

- **Use `# @package` in files** when you want the package location to be fixed regardless of how the file is loaded
- **Use `@` in defaults** when you want flexibility to place the same config in different locations
- **Avoid using both** unless you're certain about the precedence rules

### 7.9 Dot Notation for Nested Package Placement

You can use dot notation in the `@` syntax to create deeply nested structures. The format is: `@parent.child.grandchild`

#### Example: Creating Hierarchical Structures

Let's say we want to organize our configs into a multi-level hierarchy:

```yaml
#configs/config.yaml
defaults:
  - task: mnist_classification
  - task@experiments.training.models.primary: mnist_classification
  - task@experiments.training.models.backup: mnist_classification
  - task@experiments.validation.models.eval: mnist_classification
```

When we run packages.py:

```bash
experiments:
  training:
    models:
      primary:
        model:
          backbone:
            _target_: some_file.ResNet18
          adapter:
            _target_: some_file.LinearAdapter
            in_features: 512
            out_features: 10
          head:
            _target_: some_file.IdentityHead
          constant: 9
      backup:
        model:
          # ... same structure
  validation:
    models:
      eval:
        model:
          # ... same structure
task:
  model:
    # ... original task config
```

#### Practical Example: Organizing ML Experiment Config

Here's a real-world example showing how dot notation helps organize a machine learning experiment:

```yaml
#configs/config.yaml
defaults:
  - dataset@data.train: imagenet
  - dataset@data.val: imagenet
  - dataset@data.test: coco
  - model@architecture.backbone: resnet50
  - model@architecture.head: classification_head
  - optimizer@training.optim: adam
  - scheduler@training.sched: cosine
  - loss@training.criterion: cross_entropy
```

Result:

```bash
data:
  train:
    # imagenet config
  val:
    # imagenet config
  test:
    # coco config
architecture:
  backbone:
    # resnet50 config
  head:
    # classification_head config
training:
  optim:
    # adam config
  sched:
    # cosine config
  criterion:
    # cross_entropy config
```

#### Combining Dots with `# @package`

You can also use dot notation inside `# @package` directives:

```yaml
#configs/task/optimizer_settings.yaml
# @package training.hyperparameters.optimizer

learning_rate: 0.001
weight_decay: 1e-4
momentum: 0.9
```

```yaml
#configs/config.yaml
defaults:
  - task: mnist_classification
  - task/optimizer_settings
```

Output:

```bash
training:
  hyperparameters:
    optimizer:
      learning_rate: 0.001
      weight_decay: 0.0001
      momentum: 0.9
task:
  model:
    # ... task config
```

#### Benefits of Dot Notation

1. **Clear organization**: Group related configs under logical hierarchies
2. **Readability**: Easy to understand the structure from the defaults list
3. **Flexibility**: Mix and match different levels of nesting as needed
4. **Composability**: Build complex configs from simple, reusable components

### 7.10 Common Pitfalls and Best Practices

When working with packages, there are several common mistakes to avoid and best practices to follow.

#### Pitfall 1: Accidentally Overwriting Root-Level Keys with `_global_`

Using `# @package _global_` can accidentally overwrite important root-level configuration keys.

**Problem:**

```yaml
#configs/task/settings_a.yaml
# @package _global_

seed: 42
task: "This will overwrite the task config!"
```

```yaml
#configs/config.yaml
defaults:
  - task: mnist_classification  # This creates a 'task' key
  - task/settings_a              # This overwrites the 'task' key!
```

Result:

```bash
seed: 42
task: "This will overwrite the task config!"  # Original task config is gone!
```

**Solution:** Use specific package locations or be careful with key names:

```yaml
# @package _global_

seed: 42
global_task_name: "safe name that won't conflict"
```

#### Pitfall 2: Confusion Between Group Name and Package Location

The group name (folder) and package location are **different** concepts.

**Common mistake:**

```yaml
#configs/config.yaml
defaults:
  - optimizer: adam  # Group name is 'optimizer'
```

This places content at `optimizer` key by default, NOT because the folder is named "optimizer", but because that's the default package for that group.

**Understanding:**
- **Group name**: The folder path (`optimizer/adam.yaml`)
- **Package location**: Where the content goes in final config (controlled by `@` or `# @package`)

```yaml
# Same group, different package locations:
defaults:
  - optimizer: adam              # → optimizer.*
  - optimizer@training.opt: adam # → training.opt.*
  - optimizer@custom: adam       # → custom.*
```

#### Pitfall 3: Order Matters - Later Packages Override Earlier Ones

Forgetting that order matters in the defaults list leads to unexpected overrides.

**Problem:**

```yaml
#configs/config.yaml
defaults:
  - task/settings_final  # You think this is "final"
  - task/settings_draft  # But this actually wins!
  - _self_               # And this wins over everything
```

**Solution:** Always put the most important/final config **last** in the defaults list:

```yaml
defaults:
  - task/settings_draft  # Base settings
  - task/settings_final  # Override with final settings
  - _self_               # Your custom overrides (if needed)
```

#### Pitfall 4: Forgetting `_self_` is Implicit at the End

By default, `_self_` is at the end of the defaults list. This can cause confusion when you want loaded configs to override the current file.

**Problem:**

```yaml
#configs/config.yaml
defaults:
  - task/settings  # Trying to override current file's values

# But these still win because _self_ is implicitly at the end:
seed: 1000
```

**Solution:** Explicitly place `_self_` first if you want loaded configs to win:

```yaml
#configs/config.yaml
defaults:
  - _self_         # Apply current file first
  - task/settings  # Then override with settings

seed: 1000  # This will be overridden by task/settings
```

#### Pitfall 5: Not Testing Package Placement

It's easy to assume packages are placed correctly without verifying.

**Best Practice:** Always test with `--cfg job` to see the final merged config:

```bash
python packages.py --cfg job
```

Or use `--package` to inspect specific sections:

```bash
python packages.py --cfg job --package training
```

#### Pitfall 6: Mixing Relative and Absolute Paths Inconsistently

Using both `/absolute/path` and `relative/path` without understanding the difference causes confusion.

**Understanding:**
- `task/model: simple` - Relative to current config's group
- `/task/model/simple@custom` - Absolute path from configs/ root

**Best Practice:** Be consistent in your project. Use absolute paths (`/`) when you need to load configs from a specific location regardless of where the defaults list is.

#### Pitfall 7: Deep Nesting Without Necessity

Creating overly deep hierarchies makes configs hard to access.

**Problem:**

```python
learning_rate = config.experiments.training.hyperparameters.optimizer.learning_rate
```

**Solution:** Use `# @package _global_` or flatten the structure for frequently accessed values:

```yaml
# @package training

learning_rate: 0.001  # Now access as config.training.learning_rate
```

------------------------------------------------------------------------------------------------------------------------------------------

## 8. MNIST Project

A complete PyTorch Lightning project demonstrating Hydra's configuration management with a modular MNIST classification system. This example shows how grouping, instantiate, and packages work together in a real-world ML project.

### 8.1 Project Structure

```bash
├── configs
│   ├── data_module
│   │   └── mnist.yaml
│   ├── task
│   │   ├── loss_function
│   │   │   └── cross_entropy.yaml
│   │   ├── model
│   │   │   ├── adapter
│   │   │   │   └── linear_512_10.yaml
│   │   │   ├── backbone
│   │   │   │   ├── resnet18.yaml
│   │   │   │   ├── resnet34.yaml
│   │   │   │   └── resnet50.yaml
│   │   │   ├── head
│   │   │   │   └── identity_head.yaml
│   │   │   └── simple_model.yaml
│   │   ├── optimizer
│   │   │   └── adam.yaml
│   │   └── mnist_classification.yaml
│   ├── trainer
│   │   ├── cpu.yaml
│   │   └── gpu.yaml
│   └── config.yaml
├── adapters.py
├── backbones.py
├── data_modules.py
├── heads.py
├── loss_functions.py
├── models.py
├── requirements.txt
├── tasks.py
└── train.py
```

Where train.py is our main script that uses Hydra to load configs and instantiate objects.

Our config.yaml file is as such:
```yaml
#configs/config.yaml
defaults:
  - task: mnist_classification
  - data_module: mnist
  - trainer: gpu
```

Our task/mnist_classification.yaml file is as such:
```yaml
#configs/task/mnist_classification.yaml

defaults:
  - model: simple_model
  - optimizer: adam
  - loss_function: cross_entropy

_target_: tasks.MNISTClassification
```

Our task/model/simple_model.yaml file is as such:
```yaml
#configs/task/model/simple_model.yaml
# @package task.model

defaults:
  - backbone: resnet18
  - adapter: linear_512_10
  - head: identity_head

_target_: models.SimpleModel
```

Notice the `# @package task.model` directive. Without it, the content would be placed at `task.model.simple_model`, but with it, content goes directly to `task.model`.

Our task/model/backbone/resnet18.yaml file is as such:
```yaml
#configs/task/model/backbone/resnet18.yaml

_target_: backbones.ResNet18
pretrained: true
```

Our task/model/adapter/linear_512_10.yaml file is as such:
```yaml
#configs/task/model/adapter/linear_512_10.yaml

_target_: adapters.LinearAdapter
in_features: 512
out_features: 10
```

Our task/model/head/identity_head.yaml file is as such:
```yaml
#configs/task/model/head/identity_head.yaml

_target_: heads.IdentityHead
```

Our task/optimizer/adam.yaml file is as such:
```yaml
#configs/task/optimizer/adam.yaml

_target_: torch.optim.Adam
_partial_: true
lr: 5e-5
weight_decay: 0.2
```

Our task/loss_function/cross_entropy.yaml file is as such:
```yaml
#configs/task/loss_function/cross_entropy.yaml
# @package task.loss_function

_target_: loss_functions.CrossEntropyLoss
```

Our data_module/mnist.yaml file is as such:
```yaml
#configs/data_module/mnist.yaml

_target_: data_modules.MNISTDataModule
batch_size: 64
num_workers: 8
pin_memory: true
drop_last: true
data_dir: ./data/mnist
```

Our trainer/gpu.yaml file is as such:
```yaml
#configs/trainer/gpu.yaml

_target_: pytorch_lightning.Trainer

max_epochs: 10
log_every_n_steps: 10
accelerator: gpu
devices: -1
```

When we run `python train.py --cfg job`, it will print the merged config:

```bash
task:
  model:
    backbone:
      _target_: backbones.ResNet18
      pretrained: true
    adapter:
      _target_: adapters.LinearAdapter
      in_features: 512
      out_features: 10
    head:
      _target_: heads.IdentityHead
    _target_: models.SimpleModel
  optimizer:
    _target_: torch.optim.Adam
    _partial_: true
    lr: 5.0e-05
    weight_decay: 0.2
  loss_function:
    _target_: loss_functions.CrossEntropyLoss
  _target_: tasks.MNISTClassification
data_module:
  _target_: data_modules.MNISTDataModule
  batch_size: 64
  num_workers: 8
  pin_memory: true
  drop_last: true
  data_dir: ./data/mnist
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 10
  log_every_n_steps: 10
  accelerator: gpu
  devices: -1
```

### 8.2 How Configs Cascade

The config assembly follows this hierarchy:

1. **Root config** (`config.yaml`) loads 3 groups: `task`, `data_module`, `trainer`
2. **Task config** (`task/mnist_classification.yaml`) loads 3 sub-groups: `model`, `optimizer`, `loss_function` under the `task` namespace
3. **Model config** (`task/model/simple_model.yaml`) uses `# @package task.model` to place content at `task.model` (not `task.model.simple_model`), then loads 3 sub-configs: `backbone`, `adapter`, `head`

Final paths: `task.model.backbone`, `task.model.adapter`, `task.model.head`, `task.optimizer`, `task.loss_function`

### 8.3 Running Experiments

Change backbone:
```bash
python train.py task.model.backbone=resnet50
```

Use CPU trainer:
```bash
python train.py trainer=cpu
```

Override parameters:
```bash
python train.py task.optimizer.lr=0.001 data_module.batch_size=128
```

Multirun with different backbones:
```bash
python train.py -m task.model.backbone=resnet18,resnet34,resnet50
```




















