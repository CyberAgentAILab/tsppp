from logging import getLogger

import hydra

logger = getLogger(__name__)


@hydra.main(config_path="config", config_name="train_ddpm")
def main(config):
    unet = hydra.utils.instantiate(config.unet)
    logger.info("Insitantiated UNet")
    trainer = hydra.utils.instantiate(config.trainer)
    logger.info("Insitantiated Trainer")
    train_dataset = hydra.utils.instantiate(config.data)
    logger.info("Insitantiated Dataset")
    trainer.fit(unet, train_dataset)


if __name__ == "__main__":
    main()
