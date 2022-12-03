from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None)
def my_app(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    print(cfg)


if __name__ == "__main__":
    my_app()
