## 6. Instantiate

Hydra's `instantiate` function allows you to create Python objects directly from configuration files. This is a powerful feature that enables you to configure complex object hierarchies declaratively.


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
```

Output:

```bash
Hello, John!
Hello, Paul!
```

### 6.2 Partial Instantiation

The `_partial_` key creates a partial function instead of immediately instantiating the object. This is useful for classes that require parameters you don't have yet (like optimizers needing model parameters).

For example: 
```python
import torch
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=[0.1, 0.9], eps=1e-6)
print(optimizer)
```

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
    partial_optimizer = instantiate(config.optimizer)
    optimizer = partial_optimizer([parameters])
    print(optimizer)


if __name__ == "__main__":
    main()
```

Running the script will print:

```bash
python -m instantiate_06.instantiate
```

```bash
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

Run:

```bash
python -m instantiate_06.instantiate_recursive
```

Output:

```bash
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
        print(f"Config type: {type(layer_config)}")
        # You can manually instantiate it later if needed
        self.layer = instantiate(layer_config)
        print(f"Layer instantiated: {self.layer}")


@hydra.main(config_path=".", config_name="instantiate_no_recursive_config", version_base=None)
def main(config: DictConfig):
    model = instantiate(config.model)
    print(f"Model layer: {model.layer}")
```

Run it:

```bash
python -m instantiate_06.instantiate_no_recursive
```

Output:

```bash
Received config: {'_target_': 'torch.nn.Linear', 'in_features': 10, 'out_features': 5}
Config type: <class 'omegaconf.dictconfig.DictConfig'>
Layer instantiated: Linear(in_features=10, out_features=5, bias=True)
Model layer: Linear(in_features=10, out_features=5, bias=True)
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

Run:

```bash
python -m instantiate_06.instantiate_args
```

Output:

```bash
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

Output:

```bash
Received data type: <class 'dict'>
Data: {'nested': 'value', 'another_nested': {'deep': 'data'}}
obj.data type: <class 'dict'>
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
```

Output:

```bash
Got class: <class 'torch.optim.adam.Adam'>
Created optimizer: Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001
    maximize: False
    weight_decay: 0
)
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
