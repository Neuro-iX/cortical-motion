from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityd,
    ToTensord,
)


class LoadVolume(Compose):
    """Transform to load data in finetune process"""

    def __init__(self):
        super().__init__(
            [
                LoadImaged(keys="data", ensure_channel_first=True, image_only=True),
                Orientationd(keys="data", axcodes="RAS"),
                # CenterSpatialCropd(keys="data", roi_size=(160, 192, 160)),
                Resized(keys="data", spatial_size=(160, 192, 160)),
                ScaleIntensityd(keys="data", minv=0, maxv=1),
                ToTensord(keys="data", track_meta=False),
            ]
        )
