"""Module used to do inference on trained motion models."""

import os

import pandas as pd
import torch
import tqdm
from monai.data.dataset import Dataset
from sklearn.metrics import r2_score, root_mean_squared_error
from torch.utils.data import DataLoader

from src import config
from src.datasets.synthetic import SyntheticDataModule
from src.training.hyperparameters import HyperParamConf
from src.training.regression_task import RegressionTask
from src.utils.bids import BIDSDirectory
from src.utils.load import LoadVolume


def estimate_motion_dl(dl: DataLoader, model_str: str) -> list[dict[str, float]]:
    """Estimate motion using a dataloader.

    Args:
        dl (DataLoader): Dataloader containing MRI
        model_str (str): Name of model (stored in article/models)

    Returns:
        list[dict[str, float]]: list of predictions
    """
    model = (
        RegressionTask.load_from_checkpoint(
            os.path.join("article/models/", f"{model_str}.ckpt")
        )
        .cpu()
        .eval()
        .cuda()
    )

    motion_res = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dl):
            data = batch["data"].cuda()
            motions = model.predict_step(data).cpu()
            for sub, ses, path, mot in zip(
                batch["sub"], batch["ses"], batch["path"], motions
            ):
                motion_res.append(
                    {
                        "sub": sub,
                        "ses": ses,
                        "motion": mot.item(),
                        "path": path,
                    }
                )

    return motion_res


def estimate_motion_bids(dataset_dir: BIDSDirectory, model_str: str):
    """Estimate motion using a BIDSDirectory.

    Store a csv report in reports/motion_report/{dataset_name}

    Args:
        dataset_dir (BIDSDirectory): Directory to process
        model_str (str): Name of model (stored in article/models)
    """
    torch.cuda.empty_cache()
    data = []
    for sub, ses in dataset_dir.walk():
        for t1w in dataset_dir.get_all_t1w(sub, ses):
            data.append(
                {
                    "sub": sub,
                    "ses": ses,
                    "path": t1w,
                    "data": t1w,
                }
            )

    dl = DataLoader(Dataset(data, LoadVolume()), batch_size=20, num_workers=6)
    res_dict = estimate_motion_dl(dl, model_str)
    os.makedirs(f"reports/motion_report/{dataset_dir.dataset}", exist_ok=True)
    pd.DataFrame.from_records(res_dict).to_csv(
        f"reports/motion_report/{dataset_dir.dataset}/{model_str}_report.csv"
    )


def estimate_motion_test(model_str: str):
    """Estimate motion on synthetic test set.

    Store a csv report in reports/test_report/synthetic

    Args:
        model_str (str): Name of model (stored in article/models)
    """
    torch.cuda.empty_cache()
    ds = SyntheticDataModule(HyperParamConf(idx=0, batch_size=16), num_workers=6)
    ds.setup("test")
    dl = ds.test_dataloader()

    model = (
        RegressionTask.load_from_checkpoint(
            os.path.join("article/models/", f"{model_str}.ckpt")
        )
        .cpu()
        .eval()
        .cuda()
    )
    motion_res = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dl):
            data = batch["data"].cuda()
            motions = model.predict_step(data)
            for label, mot in zip(
                batch[config.HARD_LABEL_KEY],
                motions,
            ):
                motion_res.append(
                    {
                        "label": label,
                        "motion": mot.cpu(),
                    }
                )
    os.makedirs("reports/test_report/synthetic", exist_ok=True)
    df = pd.DataFrame.from_records(motion_res)
    df.to_csv(f"reports/test_report/synthetic/{model_str}_pred.csv")
    pd.DataFrame(
        {
            "model": [model],
            "r2_score": [r2_score(df.label, df.motion)],
            "rmse": [root_mean_squared_error(df.label, df.motion)],
        },
    ).to_csv(f"reports/test_report/synthetic/{model_str}_report.csv")
