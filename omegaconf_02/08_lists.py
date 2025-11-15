from omegaconf import OmegaConf
from rich import print
import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    print("="*60)
    print("1. Creating ListConfig")
    print("="*60)
    list_config = OmegaConf.create([1, 2, 3, 4, 5])
    print(f"Type: {type(list_config)}")
    print(OmegaConf.to_yaml(list_config))

    print("="*60)
    print("2. List operations - indexing and slicing")
    print("="*60)
    print(f"First element: {list_config[0]}")
    print(f"Last element: {list_config[-1]}")
    print(f"Slice [1:3]: {list_config[1:3]}")

    print("\n" + "="*60)
    print("3. Modifying lists")
    print("="*60)
    list_config[0] = 10
    list_config.append(6)
    print("After modification:")
    print(OmegaConf.to_yaml(list_config))

    print("="*60)
    print("4. Lists in configs")
    print("="*60)
    config = OmegaConf.create({
        "model": {
            "layers": [64, 128, 256, 512],
            "dropout_rates": [0.1, 0.2, 0.3]
        },
        "data": {
            "transforms": ["resize", "normalize", "augment"]
        }
    })
    print(OmegaConf.to_yaml(config))

    print("="*60)
    print("5. Accessing list elements in configs")
    print("="*60)
    print(f"First layer: {config.model.layers[0]}")
    print(f"All transforms: {config.data.transforms}")

    print("\n" + "="*60)
    print("6. List interpolation")
    print("="*60)
    interp_config = OmegaConf.create({
        "base_layers": [64, 128],
        "model": {
            "encoder_layers": "${base_layers}",
            "decoder_layers": [256, 512]
        }
    })
    print("Without resolve:")
    print(OmegaConf.to_yaml(interp_config))
    print("\nWith resolve:")
    print(OmegaConf.to_yaml(interp_config, resolve=True))

    print("\n" + "="*60)
    print("7. Converting list to Python")
    print("="*60)
    python_list = OmegaConf.to_container(list_config)
    print(f"Type: {type(python_list)}")
    print(f"Content: {python_list}")


if __name__ == "__main__":
    main()
