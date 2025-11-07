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