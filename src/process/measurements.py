import os

import pandas as pd

from src.utils.bids import BIDSDirectory


def aggregate_measurements_bids(dataset: BIDSDirectory):
    """Aggregate measurements from both hemisphere

    Args:
        dataset (BIDSDirectory): The dataset directory for aggregation
    """
    rh_path = os.path.join(dataset.base_path, "aparc_stats.rh.txt")
    lh_path = os.path.join(dataset.base_path, "aparc_stats.lh.txt")
    rh_thickness = (
        pd.read_csv(rh_path, sep="\t")
        .rename(
            columns={
                "rh_MeanThickness_thickness": "label",
                "rh.aparc.thickness": "subject",
            }
        )
        .set_index("subject")
    )
    lh_thickness = (
        pd.read_csv(lh_path, sep="\t")
        .set_index("lh.aparc.thickness")
        .rename(
            columns={
                "lh_MeanThickness_thickness": "label",
                "lh.aparc.thickness": "subject",
            }
        )
    )

    mean_thickness: pd.DataFrame = (rh_thickness["label"] + lh_thickness["label"]) / 2
    mean_thickness = mean_thickness.reset_index()
    mean_thickness["data"] = mean_thickness["subject"].apply(dataset.get_T1w)
    mean_thickness["data"] = mean_thickness["data"].str.replace(
        "/home/cbricout/scratch/HCP-D_bids/", ""
    )
    mean_thickness.to_csv(os.path.join(dataset.base_path, "participants.tsv"), sep="\t")
