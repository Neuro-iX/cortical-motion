"Core training module defining lightning logic"

import gc
import logging
import os
import shutil
from dataclasses import asdict
from enum import Enum

import comet_ml
import numpy as np
import pandas as pd
import seaborn as sb
import torch
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.tuner.tuning import Tuner
from matplotlib import pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from sklearn.metrics import r2_score, root_mean_squared_error

from src import config
from src.datasets.synthetic import SyntheticDataModule
from src.networks.generic_sfcn import GenericSFCNModel
from src.training.hyperparameters import (
    SAMPLE_TUNING,
    ClassifierType,
    HyperParamConf,
    RegressionLossType,
    TuningTask,
)
from src.utils.loss import KLDivLoss, L2Loss
from src.utils.networks import EnsureOneProcess, init_weights
from src.utils.plot import get_calibration_curve
from src.utils.slurm import get_python_slurm, launch_as_slurm
from src.utils.soft_label import ToSoftLabel


class RegressionTask(LightningModule):
    """Base LightningModule for regression task"""

    def __init__(
        self,
        hp: HyperParamConf,
        bin_range: tuple[float, float] = config.MOTION_BIN_RANGE,
    ):
        super().__init__()
        self.need_postprocess = hp.classifier != ClassifierType.VANILLA_REG
        self.need_rescale = hp.classifier == ClassifierType.VANILLA_REG
        self.kl_beta = hp.kl_beta
        self.lr = hp.lr
        self.weight_decay = hp.weight_decay
        self.beta1 = hp.beta1
        self.beta2 = hp.beta2
        self.epsilon = hp.epsilon
        self.batch_size = hp.batch_size
        self.num_bins = hp.n_bins
        self.bin_range = bin_range
        self.dropout_rate = hp.dropout

        self.model = GenericSFCNModel(hp)
        self.model.apply(init_weights)
        self.model = torch.compile(self.model)

        self.soft_label = ToSoftLabel.hp_config(hp)

        self.kl_loss = KLDivLoss(hp)
        self.l2_loss = L2Loss(hp)

        self.label: list[float] = []
        self.prediction: list[float] = []
        self.test_outputs: list[dict[str, float]] = (
            []
        )  # Container for test step outputs
        self.val_step = 0
        self.save_hyperparameters()

    def predict_step(self, batch, batch_idx=0):
        y = self.model(batch)
        return self.process_out(y)

    def process_out(self, y, cuda=False):
        """Definine postprocessing operations when using softlabels"""
        if self.need_postprocess:
            return self.soft_label.logsoft_to_hardlabel(y, cuda)
        return y.flatten()

    def compute_losses(
        self, model_out, hard_label, labels, prefix="train"
    ) -> torch.Tensor:
        """Compute losses, need to be implemented in children classes"""
        pass

    def training_step(self, batch, batch_idx=0):

        volumes = batch[config.DATA_KEY]
        labels = batch[config.LABEL_KEY]
        hard_label = batch[config.HARD_LABEL_KEY]

        if self.global_step < 6:
            for i, v in enumerate(volumes):
                self.logger.experiment.log_image(
                    image_data=v[0, :, config.VOLUME_SHAPE[2] // 2, :].cpu(),
                    name="training_sample",
                    step=(self.global_step) * len(volumes) + i,
                )

        logsoft_reg = self.model(volumes)

        gc.collect()
        train_loss = self.compute_losses(logsoft_reg, hard_label, labels, "train")
        return train_loss

    def validation_step(self, batch, batch_idx=0):

        volumes = batch[config.DATA_KEY]
        labels = batch[config.LABEL_KEY]
        hard_label = batch[config.HARD_LABEL_KEY]
        if self.val_step < 6:
            for i, v in enumerate(volumes):
                self.logger.experiment.log_image(
                    image_data=v[0, :, config.VOLUME_SHAPE[2] // 2, :].cpu(),
                    name="validation_sample",
                    step=(self.val_step) * len(volumes) + i,
                )
            self.val_step += 1
        logsoft_reg = self.model(volumes)
        val_loss = self.compute_losses(logsoft_reg, hard_label, labels, "val")

        if self.need_rescale:
            self.label += (hard_label.detach().cpu() * 4).tolist()
            self.prediction += (self.process_out(logsoft_reg.detach()) * 4).tolist()
        else:
            self.label += hard_label.detach().cpu().tolist()
            self.prediction += self.process_out(logsoft_reg.detach()).tolist()

        return val_loss

    def on_validation_epoch_end(self) -> None:

        self.log(
            "r2_score",
            r2_score(self.label, self.prediction),
        )
        dist_label, _ = np.histogram(self.label, bins=25, density=True)
        dist_pred, _ = np.histogram(self.prediction, bins=25, density=True)
        dist_label += 1e-10
        dist_pred += 1e-10
        self.log("dataset_kl", entropy(dist_label, dist_pred))
        self.log("dataset_js", jensenshannon(dist_label, dist_pred))
        self.log(
            "rmse",
            root_mean_squared_error(self.label, self.prediction),
        )
        if (
            self.logger is not None
            and hasattr(self.logger, "experiment")
            and isinstance(self.logger.experiment, comet_ml.Experiment)
        ):
            self.logger.experiment.log_figure(
                figure=get_calibration_curve(self.prediction, self.label),
                figure_name="calibration",
                step=self.global_step,
            )
            jp = sb.jointplot(
                x=self.label,
                y=self.prediction,
            )
            self.logger.experiment.log_figure(
                figure=jp.figure,
                figure_name="hist",
                step=self.global_step,
            )
        self.label = []
        self.prediction = []
        plt.close()

    def test_step(self, batch, batch_idx=0):
        volumes = batch[config.DATA_KEY]
        labels = batch[config.LABEL_KEY]
        hard_label = batch[config.HARD_LABEL_KEY]

        logits = self.model(volumes)
        predictions = self.process_out(logits)

        # Optionally, if you need to rescale predictions as in validation:
        if self.need_rescale:
            predictions = predictions * 4
            labels = hard_label * 4
        else:
            labels = hard_label

        out = {
            "predictions": predictions.detach().cpu(),
            "labels": labels.detach().cpu(),
        }
        self.test_outputs.append(out)
        return out

    def on_test_epoch_end(self):

        # Aggregate predictions and labels from all test batches:
        all_predictions = torch.cat(
            [x["predictions"] for x in self.test_outputs], dim=0
        ).numpy()
        all_labels = torch.cat([x["labels"] for x in self.test_outputs], dim=0).numpy()

        # Compute metrics:
        test_r2 = r2_score(all_labels, all_predictions)
        test_rmse = root_mean_squared_error(all_labels, all_predictions)

        self.log("test_r2", test_r2)
        self.log("test_rmse", test_rmse)

        # Compute additional distributions and divergences:
        dist_label, _ = np.histogram(all_labels, bins=25, density=True)
        dist_pred, _ = np.histogram(all_predictions, bins=25, density=True)
        dist_label += 1e-10  # Avoid division by zero
        dist_pred += 1e-10

        kl_div = entropy(dist_label, dist_pred)
        js_div = jensenshannon(dist_label, dist_pred)
        self.log("test_kl", kl_div)
        self.log("test_js", js_div)

        # Generate and log a calibration curve (assuming you have such a function):
        fig = get_calibration_curve(all_predictions, all_labels)
        self.logger.experiment.log_figure(
            figure=fig,
            figure_name="test_calibration",
            step=self.global_step,
        )

        # Optionally generate and log a scatter or joint plot:
        jp = sb.jointplot(x=all_labels, y=all_predictions)
        self.logger.experiment.log_figure(
            figure=jp.figure,
            figure_name="test_predictions_vs_labels",
            step=self.global_step,
        )

        plt.close("all")
        df = pd.DataFrame(
            {
                "predictions": all_predictions,
                "labels": all_labels,
            }
        )
        self.logger.experiment.log_table("test_pred", df)
        # Optionally clear outputs for the next test epoch
        self.test_outputs.clear()

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(self.beta1, self.beta2),
            eps=self.epsilon,
        )

        return optim


class KLRegressionTask(RegressionTask):
    """Implement Regression task for KL Divergence"""

    def compute_losses(self, model_out, hard_label, labels, prefix="train"):
        loss = self.kl_loss(model_out, labels, hard_label)
        self.log(f"{prefix}_kl_loss", loss.item(), batch_size=self.batch_size)
        self.log(f"{prefix}_loss", loss.item(), batch_size=self.batch_size)
        return loss


class L2RegressionTask(RegressionTask):
    """Implement Regression task for L2 loss"""

    def compute_losses(self, model_out, hard_label, labels, prefix="train"):
        loss = self.l2_loss(self.process_out(model_out, cuda=True), hard_label)
        self.log(f"{prefix}_l2_loss", loss.item(), batch_size=self.batch_size)
        self.log(f"{prefix}_loss", loss.item(), batch_size=self.batch_size)
        return loss


class MixedRegressionTask(RegressionTask):
    "Implement Regression task using a mix of KL and L2 loss"

    def compute_losses(self, model_out, hard_label, labels, prefix="train"):
        processed = self.process_out(model_out, cuda=True)
        l2_loss = self.l2_loss(processed, hard_label)
        kl_loss = self.kl_loss(model_out, labels, hard_label)
        loss = self.kl_beta * kl_loss + l2_loss
        self.log(
            f"{prefix}_kl_loss",
            kl_loss.item(),
            batch_size=self.batch_size,
        )
        self.log(
            f"{prefix}_l2_loss",
            l2_loss.item(),
            batch_size=self.batch_size,
        )
        self.log(f"{prefix}_loss", loss.item(), batch_size=self.batch_size)
        return loss


def launch_regression_training(hp: HyperParamConf):
    """Launch a regression experiment

    Args:
        hp (HyperParamConf): experiment's hyperparameters
    """
    torch.set_float32_matmul_precision("medium")
    torch.cuda.empty_cache()
    seed_everything(config.SEED, workers=True)
    model: RegressionTask | None = None
    if hp.loss == RegressionLossType.KL_DIV:
        model = KLRegressionTask(hp)
    elif hp.loss == RegressionLossType.L2:
        model = L2RegressionTask(hp)
    elif hp.loss == RegressionLossType.MIXED:
        model = MixedRegressionTask(hp)

    assert model is not None

    model_name = f"GenericSFCN_{hp.idx}"
    save_model_path = os.path.join("model_report", model_name)

    comet_logger = CometLogger(
        api_key=config.COMET_API_KEY,
        project_name=hp.task.value,
        experiment_name=model_name,
    )
    checkpoint = ModelCheckpoint(
        monitor="dataset_js",
        mode="min",
        save_top_k=10,
        filename="{step}-{r2_score:.2f}-{dataset_js:.3f}",
    )

    trainer = Trainer(
        max_steps=80_000,
        # max_epochs=100,
        logger=comet_logger,
        devices=torch.cuda.device_count(),
        strategy="ddp",
        accelerator="gpu",
        log_every_n_steps=100,
        val_check_interval=0.5,
        precision="16-mixed",
        sync_batchnorm=True,
        callbacks=[checkpoint],
        default_root_dir=os.path.join(config.REPORT_DIR, model_name),
    )
    if not hp.fixed_lr:
        tuner = Tuner(trainer)
        finder = tuner.lr_find(
            model,
            SyntheticDataModule(hp, 4),
            max_lr=1e-2,
            min_lr=1e-7,
            num_training=300,
        )
        if finder is not None:
            fig = finder.plot(suggest=True)
            comet_logger.experiment.log_figure(figure=fig, figure_name="learning rate")
    trainer.fit(model, datamodule=SyntheticDataModule(hp, 4))

    with EnsureOneProcess(trainer):

        logging.warning("Logging pretrain model")
        comet_logger.experiment.log_model(
            name=model_name,
            file_or_folder=checkpoint.best_model_path,
        )
        os.makedirs(save_model_path, exist_ok=True)
        shutil.copy(checkpoint.best_model_path, save_model_path)
        logging.warning("Pretrained model uploaded, saved at : %s", save_model_path)

        df = pd.DataFrame([asdict(hp)])
        r2 = 0.0
        if checkpoint.best_model_score is not None:
            r2 = checkpoint.best_model_score.item()
        df["r2_score"] = r2
        df.to_csv(os.path.join(save_model_path, "report.csv"))
        logging.warning("Report saved at : %s", save_model_path)

        logging.warning("Running test on best model checkpoint...")
        best_model = RegressionTask.load_from_checkpoint(
            checkpoint_path=checkpoint.best_model_path
        )
        trainer.test(best_model, datamodule=SyntheticDataModule(hp, 4, 16))
        logging.warning("Test completed.")


def generate_command(hp: HyperParamConf) -> str:
    """Convert a HyperParamConf object into a CLI command string."""
    parts = []
    defaults = asdict(HyperParamConf(idx=0))  # Get default values

    for param, value in asdict(hp).items():
        if value == defaults[param]:
            continue  # Skip idx and parameters matching defaults
        cli_name = f'--{param.replace("_", "-")}'
        if isinstance(value, (list, tuple)):
            str_value = ",".join(map(str, value))
        elif isinstance(value, Enum):
            str_value = value.name
        else:
            str_value = str(value)
        parts.append(f"{cli_name} {str_value}")

    return f"srun python3 cli.py train regression {' '.join(parts)}"


def tune_model(tuning_task: TuningTask):
    """Launch a tuning experiment

    Args:
        tuning_task (TuningTask): Task defining the set of
            Hyperparameters to test
    """
    for hp in SAMPLE_TUNING[tuning_task]():
        hp.task = tuning_task
        slurm = get_python_slurm(
            f"{tuning_task.name}-{hp.idx}",
            None,
            output=f"./logs/{tuning_task.name}-{hp.idx}.%j.out",
            n_cpus=10,
            n_gpus=4,
            mem="490G",
            time="15:00:00",
        )
        launch_as_slurm(slurm, generate_command(hp))
