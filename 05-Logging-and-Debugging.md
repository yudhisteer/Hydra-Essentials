## 5. Logging and Debugging

Hydra provides comprehensive logging and debugging tools to help you understand your configuration composition, track experiment outputs, and troubleshoot issues.

Hydra automatically creates output directories for each run, saves your configuration, logs your application output, and provides CLI flags to inspect configs without running your code. This makes it easy to understand what's happening and reproduce results later.

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



