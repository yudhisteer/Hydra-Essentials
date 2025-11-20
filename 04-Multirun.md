## 4. Multirun

Hydra's multirun feature allows you to run multiple experiments by sweeping over different configuration options using the `-m` or `--multirun` flag. This is essential for hyperparameter tuning, model comparison, and ablation studies.

Multirun creates a **cartesian product** of all parameter combinations. If you specify 2 experiments and 3 loss functions, you get `2 × 3 = 6` total jobs. Each job runs with a unique combination of parameters.

### 4.1 Basic Multirun Setup

We create configuration files for different experiments and loss functions in the `multirun_04/configs/` directory.

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
```

Output:

```bash
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
```

```bash
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

### 4.8 Advanced Features

**Parallel Execution:**

By default, multirun jobs execute **sequentially** (one after another). For parallel execution, you can use Hydra launcher plugins:

- **Joblib Launcher**: Local parallel execution
- **SLURM Launcher**: For HPC clusters
- **Ray Launcher**: Distributed execution
- **AWS Batch Launcher**: Cloud execution

Example with Joblib (requires `hydra-joblib-launcher` plugin(see [here](https://hydra.cc/docs/plugins/joblib_launcher/))):
```bash
python multirun_04/multirun.py -m experiment='glob(*)' \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=4
```

Note: We need to explore more about this in later sections.

-----------------------------------------------------------

