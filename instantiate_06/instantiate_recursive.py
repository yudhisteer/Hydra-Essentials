import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import warnings
warnings.filterwarnings("ignore")

from rich import print


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


if __name__ == "__main__":
    main()