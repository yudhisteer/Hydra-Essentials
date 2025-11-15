import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import warnings
warnings.filterwarnings("ignore")

from rich import print


class MyClass:
    def __init__(self, data):
        print(f"Received data type: {type(data)}")
        print(f"Data: {data}")
        self.data = data


@hydra.main(config_path=".", config_name="instantiate_convert_config", version_base=None)
def main(config: DictConfig):
    # With _convert_: all (default), data will be a regular dict
    obj = instantiate(config.my_class)
    print(f"obj.data type: {type(obj.data)}")


if __name__ == "__main__":
    main()
