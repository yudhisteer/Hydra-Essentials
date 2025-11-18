import os
from omegaconf import OmegaConf
from rich import print
import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    print("="*60)
    print("1. Loading config with interpolations")
    print("="*60)
    config = OmegaConf.load("omegaconf_02/configs/interpolation.yaml")
    print("Without resolve:")
    print(OmegaConf.to_yaml(config))

    print("\n" + "="*60)
    print("2. Resolving interpolations")
    print("="*60)
    print("With resolve=True:")
    print(OmegaConf.to_yaml(config, resolve=True))

    print("\n" + "="*60)
    print("3. Accessing interpolated values")
    print("="*60)
    print(f"Server URL (unresolved): {config.server.url}")
    # OmegaConf automatically resolves when accessing
    print(f"Client server_url: {config.client.server_url}")
    print(f"Train path: {config.paths.train}")

    print("\n" + "="*60)
    print("4. Relative interpolation with .")
    print("="*60)
    # ${.key} refers to sibling keys
    relative_config = OmegaConf.create({
        "database": {
            "host": "localhost",
            "port": 5432,
            "connection": "${.host}:${.port}"
        }
    })
    print(OmegaConf.to_yaml(relative_config, resolve=True))

    print("\n" + "="*60)
    print("5. Environment variable interpolation")
    print("="*60)
    os.environ["APP_NAME"] = "MyApp"
    os.environ["VERSION"] = "1.0.0"

    env_config = OmegaConf.create({
        "app": {
            "name": "${oc.env:APP_NAME}",
            "version": "${oc.env:VERSION}",
            "debug": "${oc.env:DEBUG,false}"  # Default value if not set
        }
    })
    print(OmegaConf.to_yaml(env_config, resolve=True))


if __name__ == "__main__":
    main()
