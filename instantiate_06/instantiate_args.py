import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import warnings
warnings.filterwarnings("ignore")

from rich import print


def my_function(pos1, pos2, keyword_arg=None):
    print(f"pos1={pos1}, pos2={pos2}, keyword_arg={keyword_arg}")


@hydra.main(config_path=".", config_name="instantiate_args_config", version_base=None)
def main(config: DictConfig):
    instantiate(config.my_function)


if __name__ == "__main__":
    main()
