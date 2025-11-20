## 3. Config Groups

Hydra's config groups allow us to organize related configurations into groups and easily switch between them. This is fundamental for managing different experiments, models, and datasets. 

Config groups are organized in subdirectories under your config path. The `subdirectory name` becomes the `group name`, and each `YAML file` in that directory is an `option` for that group. Got it?

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

Note when running the command above, by default (with no config groups in config.yaml), it will print an empty `{}`. If config groups (like `experiment`) are added via CLI or defaults, it will then print their content in YAML format.


### 3.2 Adding Config Groups from CLI

With an empty `config.yaml`, we can add config groups from the command line using the `+` prefix: 

```yaml
#grouping_03/configs/config.yaml
# Empty config
```

Run it:

```bash
python grouping_03/grouping.py +experiment=experiment_with_resnet18 hydra.job.chdir=False
```

Note: The `hydra.job.chdir=False` keeps the working directory unchanged to the one where the script is located.

Output: 

```bash
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

Instead of specifying config groups every time from the CLI, we can set defaults in `config.yaml` using the `defaults` key.

```yaml
#grouping_03/configs/config_with_defaults.yaml
defaults:
  - experiment: experiment_with_resnet18
```

Now we can run without specifying the experiment:

```bash
python grouping_03/grouping.py --config-name=config_with_defaults hydra.job.chdir=False
```

Output:

```bash
experiment:
  model: resnet18
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: adam
  scheduler: cosine
```

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
```

Output:

```bash
experiment:
  model: resnet50
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: adam
  scheduler: cosine
```

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

The `_self_` determines when values from the primary config are applied. The primary config is the current YAML file that is being used - `config_with_self.yaml` in this case.

```bash
python grouping_03/grouping.py --config-name=config_with_self hydra.job.chdir=False
```

Output:

```bash
experiment:
  model: resnet50
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: SGD      # ← This overrides the optimizer from the default (adam)
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

Note: Place `_self_` at the end so your explicit config values take precedence.

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
```

Output:

```bash
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
  - Requires `group: option` syntax - IMPORTANT
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

This organization enables swapping out models, datasets, optimizers, and other training specifics without duplicating configuration code, simply by changing the referenced file. 

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

# Global training parameters (overrides)
batch_size: 256
num_workers: 8
```

In summary, we achieve true modularity by breaking down the training configuration into small, focused components and combining them as needed to build different experiments.

### 3.9 Best Practices and Tips

#### Best Practices

**1. Use meaningful group names:**
```
Good: configs/model/, configs/optimizer/, configs/dataset/
Bad: configs/m/, configs/opt/, configs/d/
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

Note: Design your config hierarchy carefully to avoid circular dependencies.

-----------------------------------------------------

