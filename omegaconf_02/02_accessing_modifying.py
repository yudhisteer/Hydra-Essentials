from omegaconf import OmegaConf
from rich import print
import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    config = OmegaConf.load("omegaconf_02/configs/training.yaml")

    print("="*60)
    print("1. Accessing values - Dot notation")
    print("="*60)
    print(f"Batch size: {config.training.batch_size}")
    print(f"Model name: {config.model.name}")

    print("\n" + "="*60)
    print("2. Accessing values - Bracket notation")
    print("="*60)
    print(f"Learning rate: {config['training']['lr']}")
    print(f"Pretrained: {config['model']['pretrained']}")

    print("\n" + "="*60)
    print("3. Safe access with OmegaConf.select()")
    print("="*60)
    # This won't raise an error if key doesn't exist
    momentum = OmegaConf.select(config, "training.momentum", default=0.9)
    print(f"Momentum (with default): {momentum}")

    weight_decay = OmegaConf.select(config, "training.weight_decay")
    print(f"Weight decay (non-existent): {weight_decay}")

    print("\n" + "="*60)
    print("4. Modifying values")
    print("="*60)
    config.training.batch_size = 256
    config.training.lr = 0.0001
    print(f"Updated batch_size: {config.training.batch_size}")
    print(f"Updated lr: {config.training.lr}")

    print("\n" + "="*60)
    print("5. Adding new keys")
    print("="*60)
    config.training.weight_decay = 0.0001
    config.new_section = {"key": "value"}
    print("Config after adding new keys:")
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    main()
