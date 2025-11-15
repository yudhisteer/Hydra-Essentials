import hydra
from omegaconf import DictConfig
from hydra.utils import get_class
import torch
import warnings
warnings.filterwarnings("ignore")

from rich import print


@hydra.main(config_path=".", config_name="instantiate_config", version_base=None)
def main(config: DictConfig):
    # Get the class without instantiating
    optimizer_class = get_class("torch.optim.Adam")
    print(f"Got class: {optimizer_class}")

    # Use it later with your own parameters
    model = torch.nn.Linear(10, 5)
    optimizer = optimizer_class(model.parameters(), lr=0.001)
    print(f"Created optimizer: {optimizer}")


if __name__ == "__main__":
    main()
