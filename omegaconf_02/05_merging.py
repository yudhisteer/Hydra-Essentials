from omegaconf import OmegaConf
from rich import print
import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    base = OmegaConf.load("omegaconf_02/configs/base.yaml")
    override = OmegaConf.load("omegaconf_02/configs/override.yaml")

    print("="*60)
    print("Base config:")
    print("="*60)
    print(OmegaConf.to_yaml(base))

    print("="*60)
    print("Override config:")
    print("="*60)
    print(OmegaConf.to_yaml(override))

    print("="*60)
    print("1. Merging with OmegaConf.merge() - returns new config")
    print("="*60)
    merged = OmegaConf.merge(base, override)
    print("Merged config:")
    print(OmegaConf.to_yaml(merged))

    print("="*60)
    print("2. Original configs unchanged")
    print("="*60)
    print(f"Base batch_size still: {base.training.batch_size}")
    print(f"Merged batch_size: {merged.training.batch_size}")

    print("\n" + "="*60)
    print("3. Merging multiple configs")
    print("="*60)
    config1 = OmegaConf.create({"a": 1, "b": 2})
    config2 = OmegaConf.create({"b": 3, "c": 4})
    config3 = OmegaConf.create({"c": 5, "d": 6})

    merged_multi = OmegaConf.merge(config1, config2, config3)
    print("Merged result (later configs override earlier):")
    print(OmegaConf.to_yaml(merged_multi))

    print("\n" + "="*60)
    print("4. In-place merge with update()")
    print("="*60)
    base_copy = OmegaConf.load("omegaconf_02/configs/base.yaml")
    print(f"Before update - batch_size: {base_copy.training.batch_size}")

    OmegaConf.update(base_copy, "training.batch_size", 512)
    OmegaConf.update(base_copy, "model.pretrained", True)

    print(f"After update - batch_size: {base_copy.training.batch_size}")
    print(f"After update - pretrained: {base_copy.model.pretrained}")


if __name__ == "__main__":
    main()
