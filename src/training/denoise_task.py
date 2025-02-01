import gc
import logging
import os
import shutil

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CometLogger
from matplotlib import pyplot as plt
from monai.losses.ssim_loss import SSIMLoss
from sklearn.metrics import r2_score, root_mean_squared_error

from src import config
from src.datasets.synthetic import SyntheticDataModule
from src.networks.mem_unet import MemUnetModel
from src.utils.networks import EnsureOneProcess
from src.utils.plot import get_calibration_curve
from src.utils.soft_label import ToSoftLabel


class DenoiseRegressionTask(LightningModule):

    def __init__(
        self,
        lr: float = 1e-5,
        dropout_rate: float = config.DROPOUT,
        num_bins: int = config.MOTION_N_BINS,
        bin_range: tuple[float] = config.MOTION_BIN_RANGE,
    ):
        super().__init__()
        self.lr = lr
        self.num_bins = num_bins
        self.bin_range = bin_range
        self.dropout_rate = dropout_rate

        self.model = MemUnetModel(
            (1, *config.VOLUME_SHAPE), self.num_bins, self.dropout_rate
        )
        self.model = torch.compile(self.model)

        self.soft_label = ToSoftLabel.motion_config()

        self.perceptual_loss = SSIMLoss(3)
        self.pixel_loss = torch.nn.MSELoss()

        self.denoised = ()

        self.save_hyperparameters()

    def training_step(self, batch, _):
        volumes = batch[config.DATA_KEY]
        clear_vols = batch[config.CLEAR_KEY]
        batch_size = volumes.shape[0]

        denoised = self.model(volumes)
        recon_perceptual_loss = self.perceptual_loss(denoised, clear_vols)
        recon_pixel_loss = self.pixel_loss(denoised, clear_vols)

        train_loss = recon_perceptual_loss + recon_pixel_loss
        self.log("train_loss", train_loss.item(), batch_size=batch_size)
        self.log(
            "recon_perceptual_loss", recon_perceptual_loss.item(), batch_size=batch_size
        )
        self.log("recon_pixel_loss", recon_pixel_loss.item(), batch_size=batch_size)

        gc.collect()

        return train_loss

    def validation_step(self, batch, _):
        volumes = batch[config.DATA_KEY]
        clear_vols = batch[config.CLEAR_KEY]
        batch_size = volumes.shape[0]

        denoised = self.model(volumes)

        recon_perceptual_loss = self.perceptual_loss(denoised, clear_vols)
        recon_pixel_loss = self.pixel_loss(denoised, clear_vols)

        val_loss = recon_perceptual_loss + recon_pixel_loss
        self.log("val_loss", val_loss.item(), batch_size=batch_size)
        self.log(
            "val_recon_perceptual_loss",
            recon_perceptual_loss.item(),
            batch_size=batch_size,
        )
        self.log("val_recon_pixel_loss", recon_pixel_loss.item(), batch_size=batch_size)

        self.denoised = (
            denoised.detach().cpu()[0, 0, :, config.VOLUME_SHAPE[2] // 2, :],
            clear_vols.detach().cpu()[0, 0, :, config.VOLUME_SHAPE[2] // 2, :],
            volumes.detach().cpu()[0, 0, :, config.VOLUME_SHAPE[2] // 2, :],
        )
        return val_loss

    def on_validation_epoch_end(self) -> None:
        self.logger.experiment.log_image(
            image_data=self.denoised[0], name="denoised_image", step=self.global_step
        )
        self.logger.experiment.log_image(
            image_data=self.denoised[1], name="clear_image", step=self.global_step
        )
        self.logger.experiment.log_image(
            image_data=self.denoised[2], name="noisy_image", step=self.global_step
        )

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        return optim


def launch_denoise_training():
    torch.cuda.empty_cache()
    model = DenoiseRegressionTask(
        1e-4,
    )
    save_model_path = os.path.join("model_report", "narval-run-denoise")

    comet_logger = CometLogger(
        api_key=config.COMET_API_KEY,
        project_name="cortical",
        experiment_name="narval-run-denoise",
    )

    trainer = Trainer(
        max_epochs=10,
        logger=comet_logger,
        devices=torch.cuda.device_count(),
        strategy="ddp",
        accelerator="gpu",
        log_every_n_steps=10,
        val_check_interval=100,
        precision="16-mixed",
        sync_batchnorm=True,
    )

    trainer.fit(model, datamodule=SyntheticDataModule(8, 5))

    with EnsureOneProcess(trainer):

        logging.warning("Logging pretrain model")
        comet_logger.experiment.log_model(
            name="ImprovedUnetModel",
            file_or_folder=checkpoint.best_model_path,
        )
        shutil.copy(checkpoint.best_model_path, save_model_path)
        logging.warning("Pretrained model uploaded, saved at : %s", save_model_path)
        logging.info("Removing Checkpoints")
        shutil.rmtree(trainer.default_root_dir)
