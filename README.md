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
