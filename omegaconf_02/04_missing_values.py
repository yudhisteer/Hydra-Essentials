from omegaconf import OmegaConf, MissingMandatoryValue
from rich import print
import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    config = OmegaConf.load("omegaconf_02/configs/missing_values.yaml")

    print("="*60)
    print("1. Config with missing mandatory values (???)")
    print("="*60)
    print(OmegaConf.to_yaml(config))

    print("\n" + "="*60)
    print("2. Checking if value is missing")
    print("="*60)
    print(f"Is username missing? {OmegaConf.is_missing(config, 'database.username')}")
    print(f"Is host missing? {OmegaConf.is_missing(config, 'database.host')}")

    print("\n" + "="*60)
    print("3. Attempting to access missing value (will raise error)")
    print("="*60)
    try:
        username = config.database.username
        print(f"Username: {username}")
    except MissingMandatoryValue as e:
        print(f"Error: {e}")

    print("\n" + "="*60)
    print("4. Setting mandatory values")
    print("="*60)
    config.database.username = "admin"
    config.database.password = "secret123"
    print("After setting values:")
    print(OmegaConf.to_yaml(config))

    print("\n" + "="*60)
    print("5. Accessing values now works")
    print("="*60)
    print(f"Username: {config.database.username}")
    print(f"Password: {config.database.password}")


if __name__ == "__main__":
    main()
