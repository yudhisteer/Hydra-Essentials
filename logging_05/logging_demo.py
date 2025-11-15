import logging
import warnings
from omegaconf import OmegaConf, DictConfig
import hydra
from rich import print

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    logger.info("INFO: Starting training...")
    logger.debug("DEBUG: Printing config...")
    logger.warning("WARNING: This is a warning message")

    print("\n" + "="*50)
    print("Current Configuration:")
    print("="*50)
    print(OmegaConf.to_yaml(config))

    logger.info(f"INFO: Training with batch_size={config.training.batch_size}")
    logger.info(f"INFO: Model: {config.model.name}")
    logger.info("INFO: Training completed!")


if __name__ == "__main__":
    main()
