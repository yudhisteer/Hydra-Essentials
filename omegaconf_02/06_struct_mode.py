from omegaconf import OmegaConf
from rich import print
import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    print("="*60)
    print("1. Default behavior - can add new keys")
    print("="*60)
    config = OmegaConf.create({"model": {"name": "resnet18"}})
    config.model.layers = 18  # This works
    config.new_key = "new_value"  # This also works
    print(OmegaConf.to_yaml(config))

    print("\n" + "="*60)
    print("2. Struct mode - prevents adding new keys")
    print("="*60)
    struct_config = OmegaConf.create({"model": {"name": "resnet18"}})
    OmegaConf.set_struct(struct_config, True)

    # This works - modifying existing key
    struct_config.model.name = "resnet50"
    print(f"Modified existing key: {struct_config.model.name}")

    # This will raise an error - adding new key
    print("\nAttempting to add new key in struct mode...")
    try:
        struct_config.model.layers = 18
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

    print("\n" + "="*60)
    print("3. Disabling struct mode")
    print("="*60)
    OmegaConf.set_struct(struct_config, False)
    struct_config.model.layers = 18  # Now this works
    print("After disabling struct mode:")
    print(OmegaConf.to_yaml(struct_config))

    print("\n" + "="*60)
    print("4. Read-only mode")
    print("="*60)
    readonly_config = OmegaConf.create({"model": {"name": "resnet18"}})
    OmegaConf.set_readonly(readonly_config, True)

    print(f"Is read-only? {OmegaConf.is_readonly(readonly_config)}")

    # This will raise an error
    print("\nAttempting to modify read-only config...")
    try:
        readonly_config.model.name = "resnet50"
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

    print("\n" + "="*60)
    print("5. Converting to Python dict with to_container()")
    print("="*60)
    config = OmegaConf.create({
        "model": {"name": "resnet18", "layers": 18},
        "training": {"batch_size": 128}
    })

    # Convert to plain Python dict
    python_dict = OmegaConf.to_container(config)
    print(f"Type: {type(python_dict)}")
    print(f"Content: {python_dict}")

    # Can use as regular dict
    python_dict["new_key"] = "works"
    print(f"After adding key to dict: {python_dict}")


if __name__ == "__main__":
    main()
