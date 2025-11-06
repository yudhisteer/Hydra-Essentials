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