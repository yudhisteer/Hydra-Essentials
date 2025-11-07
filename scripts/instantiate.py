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