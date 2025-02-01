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
from src.networks.improved_unet import ImprovedUnetModel
from src.networks.simple_unet import SUnetModel, SUnetModelMap
from src.utils.loss import KLDivLoss
from src.utils.networks import EnsureOneProcess, init_weights
from src.utils.plot import get_calibration_curve
from src.utils.soft_label import ToSoftLabel


class DenoiseRegressionTask(LightningModule):

    def __init__(
        self,
        model: str,
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

        if model == "SUnet":
            self.model = SUnetModel(
                (1, *config.VOLUME_SHAPE), self.num_bins, self.dropout_rate
            )
        elif model == "ImUnet":
            self.model = ImprovedUnetModel(
                (1, *config.VOLUME_SHAPE), self.num_bins, self.dropout_rate
            )
        elif model == "SUnetMap":
            self.model = SUnetModelMap(
                (1, *config.VOLUME_SHAPE), self.num_bins, self.dropout_rate
            )
        self.model.apply(init_weights)
        self.model = torch.compile(self.model)

        self.soft_label = ToSoftLabel.motion_config()

        self.label_loss = KLDivLoss()
        self.perceptual_loss = SSIMLoss(3)
        self.pixel_loss = torch.nn.L1Loss()

        self.label = []
        self.prediction = []
        self.denoised = ()

        self.save_hyperparameters()

    def process_logsoft(self, y):
        return self.soft_label.logsoft_to_hardlabel(y)

    def training_step(self, batch, _):
        volumes = batch[config.DATA_KEY]
        labels = batch[config.LABEL_KEY]
        clear_vols = batch[config.CLEAR_KEY]
        batch_size = labels.shape[0]

        denoised, logsoft_reg, _ = self.model(volumes)
        label_loss = self.label_loss(logsoft_reg, labels)
        recon_perceptual_loss = self.perceptual_loss(denoised, clear_vols)
        recon_pixel_loss = self.pixel_loss(denoised, clear_vols)
        # denoise_loss = denoise.abs().mean()

        recon_loss = recon_perceptual_loss + recon_pixel_loss

        train_loss = label_loss + recon_loss
        self.log("train_loss", train_loss.item(), batch_size=batch_size)
        self.log("recon_loss", recon_loss.item(), batch_size=batch_size)

        self.log(
            "recon_perceptual_loss", recon_perceptual_loss.item(), batch_size=batch_size
        )
        self.log("recon_pixel_loss", recon_pixel_loss.item(), batch_size=batch_size)
        # self.log("denoise_loss", denoise_loss.item(), batch_size=batch_size)
        self.log("label_loss", label_loss.item(), batch_size=batch_size)

        gc.collect()

        return train_loss

    def validation_step(self, batch, _):
        volumes = batch[config.DATA_KEY]
        labels = batch[config.LABEL_KEY]
        clear_vols = batch[config.CLEAR_KEY]
        hard_label = batch[config.HARD_LABEL_KEY].detach().cpu()

        batch_size = labels.shape[0]

        denoised, logsoft_reg, _ = self.model(volumes)

        label_loss = self.label_loss(logsoft_reg, labels)
        recon_perceptual_loss = self.perceptual_loss(denoised, clear_vols)
        recon_pixel_loss = self.pixel_loss(denoised, clear_vols)
        # denoise_loss = denoise.abs().mean()

        recon_loss = recon_perceptual_loss + recon_pixel_loss
        val_loss = label_loss + recon_loss
        self.log("val_loss", val_loss.item(), batch_size=batch_size)
        self.log("val_recon_loss", recon_loss.item(), batch_size=batch_size)

        self.log(
            "val_recon_perceptual_loss",
            recon_perceptual_loss.item(),
            batch_size=batch_size,
        )
        self.log("val_recon_pixel_loss", recon_pixel_loss.item(), batch_size=batch_size)
        # self.log("val_denoise_loss", denoise_loss.item(), batch_size=batch_size)

        self.log("val_label_loss", label_loss.item(), batch_size=batch_size)

        self.label += hard_label.tolist()
        self.prediction += self.process_logsoft(logsoft_reg.detach().cpu()).tolist()
        self.denoised = (
            denoised.detach().cpu()[0, 0, :, config.VOLUME_SHAPE[2] // 2, :],
            clear_vols.detach().cpu()[0, 0, :, config.VOLUME_SHAPE[2] // 2, :],
            volumes.detach().cpu()[0, 0, :, config.VOLUME_SHAPE[2] // 2, :],
            # denoise.detach().cpu()[0, 0, :, config.VOLUME_SHAPE[2] // 2, :],
        )
        return val_loss

    def on_validation_epoch_end(self) -> None:
        self.log(
            "r2_score",
            r2_score(self.label, self.prediction),
            sync_dist=True,
        )

        self.log(
            "rmse",
            root_mean_squared_error(self.label, self.prediction),
            sync_dist=True,
        )
        self.logger.experiment.log_figure(
            figure=get_calibration_curve(self.prediction, self.label),
            figure_name="calibration",
            step=self.global_step,
        )
        self.logger.experiment.log_image(
            image_data=self.denoised[0], name="denoised_image", step=self.global_step
        )
        self.logger.experiment.log_image(
            image_data=self.denoised[1], name="clear_image", step=self.global_step
        )
        self.logger.experiment.log_image(
            image_data=self.denoised[2], name="noisy_image", step=self.global_step
        )
        # self.logger.experiment.log_image(
        #     image_data=self.denoised[3], name="denoise_map", step=self.global_step
        # )

        self.label = []
        self.prediction = []
        plt.close()

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)

        return optim


def launch_training(model_name: str):
    torch.set_float32_matmul_precision("medium")
    torch.cuda.empty_cache()
    model = DenoiseRegressionTask(
        model_name,
        5e-5,
    )
    save_model_path = os.path.join("model_report", model_name)

    comet_logger = CometLogger(
        api_key=config.COMET_API_KEY,
        project_name="combine",
        experiment_name=model_name,
    )
    checkpoint = ModelCheckpoint(monitor="r2_score", mode="max")

    trainer = Trainer(
        max_epochs=6,
        logger=comet_logger,
        devices=torch.cuda.device_count(),
        strategy="ddp",
        accelerator="gpu",
        log_every_n_steps=10,
        val_check_interval=500,
        precision="16-mixed",
        sync_batchnorm=True,
        callbacks=[checkpoint],
    )

    trainer.fit(model, datamodule=SyntheticDataModule(6, 6, 12))

    with EnsureOneProcess(trainer):

        logging.warning("Logging pretrain model")
        comet_logger.experiment.log_model(
            name=model_name,
            file_or_folder=checkpoint.best_model_path,
        )
        shutil.copy(checkpoint.best_model_path, save_model_path)
        logging.warning("Pretrained model uploaded, saved at : %s", save_model_path)
        logging.info("Removing Checkpoints")
        shutil.rmtree(trainer.default_root_dir)
