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

OmegaConf wraps Python dicts and lists in `DictConfig` and `ListConfig` objects, providing additional functionality like `interpolation`, `validation`, and `type checking`.

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

