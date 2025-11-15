from omegaconf import OmegaConf
from rich import print
import os
import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    print("="*60)
    print("1. Creating a config")
    print("="*60)
    config = OmegaConf.create({
        "model": {
            "name": "resnet50",
            "pretrained": True,
            "layers": 50
        },
        "training": {
            "batch_size": 128,
            "epochs": 100,
            "lr": 0.001
        }
    })
    print(OmegaConf.to_yaml(config))

    print("="*60)
    print("2. Saving config to file")
    print("="*60)
    output_path = "omegaconf_02/saved_config.yaml"
    OmegaConf.save(config, output_path)
    print(f"Config saved to: {output_path}")

    print("\n" + "="*60)
    print("3. Loading saved config")
    print("="*60)
    loaded = OmegaConf.load(output_path)
    print("Loaded config:")
    print(OmegaConf.to_yaml(loaded))

    print("="*60)
    print("4. Verifying configs are equal")
    print("="*60)
    print(f"Configs are equal: {config == loaded}")

    # Clean up
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"\nCleaned up: {output_path}")


if __name__ == "__main__":
    main()
