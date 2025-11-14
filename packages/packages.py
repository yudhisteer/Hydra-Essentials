import hydra
from omegaconf import DictConfig, OmegaConf

import warnings
warnings.filterwarnings("ignore")


@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

if __name__ == "__main__":
    main()