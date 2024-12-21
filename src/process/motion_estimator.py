"""This module use the model trained in the IBSI25 Paper to estimate amounts of motion"""

import os

import pandas as pd
import torch
import tqdm
from monai.data.dataset import Dataset
from torch.utils.data import DataLoader

from src import config
from src.utils import bids
from src.utils.bids import BIDSDirectory
from src.utils.load import LoadVolume
from src.utils.soft_label import ToSoftLabel


def estimate_motion_dl(dl: DataLoader) -> list[dict[str, float]]:
    model = torch.load(config.MOTION_MODEL_PATH).cuda()
    soft_tsf = ToSoftLabel.motion_config()
    motion_res = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dl):
            data = batch["data"].cuda()
            motions = model(data)
            for sub, ses, path, mot in zip(
                batch["sub"], batch["ses"], batch["path"], motions
            ):
                motion_res.append(
                    {
                        "sub": sub,
                        "ses": ses,
                        "motion": soft_tsf.logsoft_to_hardlabel(mot).item(),
                        "path": path,
                    }
                )

    return motion_res


def estimate_motion_bids(dataset_dir: BIDSDirectory):
    data = []
    for sub, ses in dataset_dir.walk():
        for t1w in dataset_dir.get_all_T1w(sub, ses):
            data.append(
                {
                    "sub": sub,
                    "ses": ses,
                    "path": t1w,
                    "data": t1w,
                }
            )
    dl = DataLoader(Dataset(data, LoadVolume()), batch_size=16)
    res_dict = estimate_motion_dl(dl)
    os.makedirs("reports/motion_report", exist_ok=True)
    pd.DataFrame.from_records(res_dict).to_csv("reports/motion_report/report.csv")


def estimate_motion_records(record_path: str):
    records = pd.read_csv(record_path)
    records["sub"] = records["data"].apply(bids.get_sub)
    records["ses"] = records["data"].apply(bids.get_ses)
    records["path"] = records["data"]

    dl = DataLoader(Dataset(records.to_dict("records"), LoadVolume()), batch_size=16)
    res_dict = estimate_motion_dl(dl)
    os.makedirs("reports/motion_report", exist_ok=True)
    pd.DataFrame.from_records(res_dict).to_csv("reports/motion_report/report.csv")
