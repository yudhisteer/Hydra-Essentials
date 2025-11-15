from omegaconf import OmegaConf
from rich import print
import sys
import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    print("="*60)
    print("OmegaConf CLI Example")
    print("="*60)
    print(f"Command line args: {sys.argv[1:]}\n")

    # Parse CLI arguments
    config = OmegaConf.from_cli()

    print("Parsed configuration:")
    print(OmegaConf.to_yaml(config))

    # Load base config and merge with CLI
    if len(sys.argv) > 1:
        print("\n" + "="*60)
        print("Merging with base config")
        print("="*60)
        base = OmegaConf.load("omegaconf_02/configs/training.yaml")
        merged = OmegaConf.merge(base, config)
        print("Merged result:")
        print(OmegaConf.to_yaml(merged))


if __name__ == "__main__":
    main()
