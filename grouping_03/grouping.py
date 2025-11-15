from omegaconf import OmegaConf, DictConfig
import hydra
from rich import print
import warnings
warnings.filterwarnings("ignore")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    main()
