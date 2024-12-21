import logging

from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityd,
    ToTensord,
    Transform,
)


class LogData(Transform):
    def __init__(self, keys="data"):
        self.keys = keys

    def __call__(self, data):
        logging.info(data[self.keys])
        return data


class LoadVolume(Compose):
    """Transform to load data in finetune process"""

    def __init__(self):
        super().__init__(
            [
                LogData(),
                LoadImaged(keys="data", ensure_channel_first=True, image_only=True),
                Orientationd(keys="data", axcodes="RAS"),
                CenterSpatialCropd(keys="data", roi_size=(160, 192, 160)),
                Resized(keys="data", spatial_size=(160, 192, 160)),
                ScaleIntensityd(keys="data", minv=0, maxv=1),
                ToTensord(keys="data", track_meta=False),
            ]
        )
