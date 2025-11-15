from omegaconf import OmegaConf
from rich import print
import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    print("="*60)
    print("1. Creating config from Python dict")
    print("="*60)
    config = OmegaConf.create({
        "training": {
            "batch_size": 128,
            "epochs": 30,
            "lr": 0.001
        },
        "model": {
            "name": "resnet18",
            "pretrained": True
        }
    })
    print(OmegaConf.to_yaml(config))

    print("\n" + "="*60)
    print("2. Creating config from list")
    print("="*60)
    config_list = OmegaConf.create([1, 2, 3, "four", {"key": "value"}])
    print(OmegaConf.to_yaml(config_list))

    print("\n" + "="*60)
    print("3. Creating from dotlist")
    print("="*60)
    dotlist = ["training.batch_size=256", "training.lr=0.0001", "model.name=resnet50"]
    config_from_dotlist = OmegaConf.from_dotlist(dotlist)
    print(OmegaConf.to_yaml(config_from_dotlist))


if __name__ == "__main__":
    main()
