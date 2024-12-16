import torch
from src.datasets.common import BaseDataset
import src.config as config


class HCPDev(BaseDataset):
    dataset_dir = config.HCPDEV_FOLDER
    csv_path = "participants.tsv"
    volumes = ["data"]
    labels = ["label"]

    def prepare_label(
        self, lab_key: str, lab_value: str
    ) -> torch.IntTensor | torch.FloatTensor:

        return torch.FloatTensor(lab_value)
