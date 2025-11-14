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

------------------------------------------------------------------------------------------------------------------------------------

## 2. OmegaConf

In this section we will learn how to use OmegaConf to create and manipulate config files.

```bash
#scripts/02_omegaconf.py
from omegaconf import OmegaConf

from rich import print
import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    config = OmegaConf.create({"training": {"batch_size": 128, "nrof_epochs": 30, "lr": 5e-3}})
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    main()
```

This will print:

```bash
training:
  batch_size: 128
  nrof_epochs: 30
  lr: 0.005
```

Now insead of using the `.create()` method, we can use the `.load()` method:

```bash
#scripts/02_omegaconf.py
from omegaconf import OmegaConf

from rich import print
import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    config = OmegaConf.load("./scripts/config.yaml")
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    main()
```

Next we can use the `.from_dotlist()` method to create a config from a list of dot-separated strings.

```bash
#scripts/02_omegaconf.py
from omegaconf import OmegaConf

from rich import print
import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    my_items = ["training.batch_size=128", "training.nrof_epochs=30", "training.lr=5e-3"]
    config = OmegaConf.from_dotlist(my_items)
    print(OmegaConf.to_yaml(config))
    

if __name__ == "__main__":
    main()
```

This will print:

```bash
╰─$ python 'scripts/02_omegaconf.py'                                                                                                                   1 ↵
training:
  batch_size: 128
  nrof_epochs: 30
  lr: 0.005
```

We can also use CLI using the `.from_cli()` method.

```bash
#scripts/02_omegaconf.py
from omegaconf import OmegaConf

from rich import print
import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    config = OmegaConf.from_cli()
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    main()
```

Using command:

```bash
╰─$ python 'scripts/02_omegaconf.py'  training.nrof_epochs=50 training.optimizer=adam
training:
  nrof_epochs: 50
  optimizer: adam
```

We can also put some variables to be mandatory using "???" in the config file.

```yaml
#scripts/config.yaml
training:
  batch_size: 256
  nrof_epochs: 30
  lr: 0.005
  scheduler: ???
```

```python
#scripts/02_omegaconf.py
from omegaconf import OmegaConf

from rich import print
import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    config = OmegaConf.load("./scripts/config.yaml")

    # different ways of printing
    print(config.training.batch_size)
    print(config["training"]["lr"])

    # updating configs
    config.training.batch_size = 256

    # create new configs
    config.model = {"name": "resnet18"}

    # choosing the scheduler which is mandatory
    config.training.scheduler = "cosine"

    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    main()
```

This will print:

```bash
training:
  batch_size: 256
  nrof_epochs: 30
  lr: 0.005
  scheduler: cosine
model:
  name: resnet18
```

We can also use `resolve=True` to resolve the interpolated values in the config file.

```yaml
#scripts/interpolated_config.yaml
server:
  host: localhost
  port: 8080

client:
  url: ${server.host}:${server.port}
  timeout: 10
  description: Client of ${.url}
```


```python
#scripts/02_omegaconf.py
from omegaconf import OmegaConf

from rich import print
import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    config = OmegaConf.load("./scripts/interpolated_config.yaml")
    print(OmegaConf.to_yaml(config, resolve=True))


if __name__ == "__main__":
    main()
```

This will print:

```bash
╰─$ python scripts/02_omegaconf.py
server:
  host: localhost
  port: 8080
client:
  url: localhost:8080
  timeout: 10
  description: Client of localhost:8080
```

We can also have an environment variable in the config file.

```yaml
#scripts/env_config.yaml
user:
  name: ${oc.env:USER}
  password: ${oc.env:PASSWORD, default_password}
```

```python
#scripts/02_omegaconf.py
import os

from omegaconf import OmegaConf
from rich import print
import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    config = OmegaConf.load("./scripts/env_config.yaml")
    os.environ["USER"] = "yoyo"
    os.environ["PASSWORD"] = "123456"

    print(OmegaConf.to_yaml(config, resolve=True))


if __name__ == "__main__":
    main()
```

This will print:

```bash
user:
  name: yoyo
  password: 123456
```

Finally, we can also use the `.merge()` method to merge two config files.

```python
#scripts/02_omegaconf.py
from omegaconf import OmegaConf

from rich import print
import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    config = OmegaConf.load("./scripts/env_config.yaml")
    config2 = OmegaConf.load("./scripts/config.yaml")
    config.merge(config2)
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    main()
```

This will print:

```bash
user:
  name: yoyo
  password: 123456
training:
  batch_size: 256
  nrof_epochs: 30
  lr: 0.005
  scheduler: cosine
model:
  name: resnet18
```

## 3. Grouping

We can also group configs into a single config file. We start by creatng two config file as shown below:

```yaml
#configs/experiments/experiment_with_resnet18.yaml
model: resnet18
epochs: 100
batch_size: 128
lr: 0.001
optimizer: adam
scheduler: cosine
```

```yaml
#configs/experiments/experiment_with_resnet50.yaml
model: resnet50
epochs: 100
batch_size: 128
lr: 0.001
optimizer: adam
scheduler: cosine
```

In our code we need to specfy the empty config file ouside the experiments/ dir.

```python
#scripts/03_grouping.py
from omegaconf import OmegaConf, DictConfig
import hydra
from rich import print
import warnings
warnings.filterwarnings("ignore")


@hydra.main(config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    main()
```

Now we can run out file using `+experiment=experiment_with_resnet18` and this will overwrite our empty config.yaml file in he outputs/ dir. 

```bash
╰─$ python scripts/03_grouping.py +experiment=experiment_with_resnet18
experiments:
  model: resnet18
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: adam
  scheduler: cosine
```

Before our `config.yaml` file was empty but we can configure it to have a **default** experiment. like this:

```yaml
#configs/config.yaml
defaults:
  - experiment: experiment_with_resnet18
```

Now we can run our file without specifying the experiment.

```bash
╰─$ python scripts/03_grouping.py
experiments:
  model: resnet18
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: adam
  scheduler: cosine
```

We can also load other experiments using:

```yaml
#configs/config.yaml
defaults:
  - experiment: experiment_with_resnet18
  - override experiment: experiment_with_resnet50 # This will override the default experiment
```

This will print:

```bash
╰─$ python scripts/03_grouping.py
experiment:
  model: resnet50
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: adam
  scheduler: cosine
```

We can also override the default experiment using:

```yaml
#configs/config.yaml
defaults:
  - experiment: experiment_with_resnet18
  - override experiment: experiment_with_resnet50
  - _self_

experiment:
  optimizer: SGD
```

We use `- _self_` to override the default experiment.

```bash
╰─$ python scripts/03_grouping.py
experiment:
  model: resnet50
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: SGD <---- This is the overridden optimizer
  scheduler: cosine
```

We can also merge another config file into the main config file. We creete a new confg file named `demo_config.yaml` as shown below:

```yaml
#configs/demo_config.yaml
seed: 42
```
We then specify the name of the config file to merge in the `config.yaml` file.

```yaml
#configs/config.yaml
defaults:
  - experiment: experiment_with_resnet18
  - demo_config <---- another config file name to merge
  - _self_

experiment:
  optimizer: SGD
```

```bash
╰─$ python scripts/03_grouping.py                                                                                                                                                  1 ↵
experiment:
  model: resnet18
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: SGD
  scheduler: cosine
seed: 42
```

## 4. Multirun

We can use the `-m` flag to run multiple experiments at once. We create new .yaml file in the `configs/loss_function` folder as shown below:

```yaml
#configs/loss_function/softmax.yaml
name: softmax
```

```yaml
#configs/loss_function/cosface.yaml
name: cosface
margin: 0.5
```

```yaml
#configs/loss_function/arcface.yaml
name: arcface
margin: 0.8
```

Now we can run the experiments using the `-m` flag as shown below. Sicne we have 2 experiments and 3 loss functions, we will have 6 jobs in total.

```bash
╰─$ python scripts/03_grouping.py -m experiment=experiment_with_resnet18,experiment_with_resnet50 loss_function=arcface,cosface,softmax
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

To simplify the CLI command, we can use the `glob` syntax. Notice that we use `exclude` to exclude the softmax loss function.

```bash
─$ python scripts/03_grouping.py -m experiment='glob(*)' loss_function='glob(*, exclude=soft*)'                                       
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


## 5. Logging and Debugging

We can specify we want to loging or not tin the config file. If we choose "default" or "stdout", it will create a log file in the outputs/ dir.

```yaml
#configs/config.yaml
defaults:
  - experiment: experiment_with_resnet18
  - loss_function: arcface
  - demo_config
  - _self_
  - override hydra/job_logging: none #choose between none, disabled, default or stdout


experiment:
  optimizer: SGD

hydra:
  verbose: True
```

This is the file we creae in the outputs/ dir.

```bash
#outputs/{timestamp}/{timestamp}script.log
[2025-11-06 17:28:23,017][__main__][INFO] - INFO: Printing config...
[2025-11-06 17:28:23,017][__main__][DEBUG] - DEBUG: Printing optimizer...
```

But if we choose "none", it will not create a log file but still print the logs to the console until verbose is set to True.

```python
#scripts/03_grouping.py
import logging
import warnings

from omegaconf import OmegaConf, DictConfig
import hydra
from rich import print
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    logger.info("INFO: Printing config...")
    logger.debug("DEBUG: Printing optimizer...")


if __name__ == "__main__":
    main()
```

And this is the output in the console.

```bash
...
loss_function:
  name: arcface
  margin: 0.8
seed: 42

[2025-11-06 17:26:27,835][HYDRA] INFO: Printing config...
[2025-11-06 17:26:27,835][HYDRA] DEBUG: Printing optimizer...
```

In order to debug our config file, we can use the `--cfg job` flag. This will print the config file. Note that we are not using the print function in the script. Our `main` function could have a `pass` statement and it will still print the config file.

```bash
╰─$ python '/home/yoyo/Hydra-Essentials/scripts/03_grouping.py' --cfg job
experiment:
  model: resnet18
  epochs: 100
  batch_size: 128
  lr: 0.001
  optimizer: SGD
  scheduler: cosine
loss_function:
  name: arcface
  margin: 0
```

We can also only specify the package we want to debug usng the `--package` flag. This is helful when we have a lot of packages and we only want to debug one of them.

```bash
╰─$ python '/home/yoyo/Hydra-Essentials/scripts/03_grouping.py' --cfg job --package experiment
# @package experiment
model: resnet18
epochs: 100
batch_size: 128
lr: 0.001
optimizer: SGD
```

## 6. Instantiate

We can instantiate a class using hydra. This is useful when we want to instantiate a class with a config file. We use the `_target_` key to specify the class to instantiate and the `name` key to pass the name to the class.
We can also use the `hydra.utils.instantiate` function to instantiate the class.

```yaml
#scripts/instantiate_config.yaml
my_class:
  # we are specifying the class to instantiate
  _target_: scripts.instantiate.MyClass
  # we are passing the name to the class
  name: Paul
```

```python
#scripts/instantiate.py
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

Running the script as a module wih  `-m` flag will print:  

```bash
╰─$ python -m scripts.instantiate
Hello, John!
Hello, Paul!
```

Now what if we want to instantiate a class which depends on other classes? We can use the `_partial_` key to tell hydra that this is a partial config and it will not raise an error if the config is not complete.

```yaml
#scripts/instantiate_config.yaml
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

We create the partial optimizer by passing the parameters to the optimizer class. Note we could have not passed the parameters to the optimizer class and it would have been instantiated with the default parameters.

```python
#scripts/instantiate.py
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
╰─$ python -m scripts.instantiate
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























