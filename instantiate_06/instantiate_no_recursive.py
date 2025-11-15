import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import warnings
warnings.filterwarnings("ignore")

from rich import print


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


if __name__ == "__main__":
    main()
