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