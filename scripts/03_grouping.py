import logging
import warnings

from omegaconf import OmegaConf, DictConfig
import hydra
from rich import print
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    logger.info("INFO: Printing config...")
    logger.debug("DEBUG: Printing optimizer...")


if __name__ == "__main__":
    main()