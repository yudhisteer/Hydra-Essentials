## 8. MNIST Project

A complete project demonstrating Hydra's configuration management with a modular MNIST classification system. This example shows how grouping, instantiate, and packages work together in a real-world ML project.

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

Here again, the `# @package task.loss_function` directive is used to place the content at `task.loss_function`.
Without it, the content would be placed at `task.loss_function.cross_entropy`.

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

When we run `python mnist_project_08/train.py --cfg job`, it will print the merged config:

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




















