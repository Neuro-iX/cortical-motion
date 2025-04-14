from monai.transforms import OneOf, RandHistogramShift, Randomizable
from torchio.transforms import RandomBlur, RandomFlip, RandomGamma, RandomNoise


class AugmentMRI(Randomizable):
    def __init__(
        self,
        volume_key,
        hist_shift=False,
    ):
        super().__init__()
        self.volume_key = volume_key
        self.hist_shift = hist_shift

        self.flip = RandomFlip(axes=0)
        self.blur = RandomBlur((0, 0.05))
        self.noise = RandomNoise(std=(0, 0.1))
        self.base_tsf = (
            self.noise,
            self.blur,
        )
        self.transform = OneOf(self.base_tsf)

        self.contrast_transform = OneOf(
            (RandHistogramShift((10, 15), prob=1), RandomGamma((-0.15, 0.15)))
        )

    def randomize(self, data):
        self.apply_augment = self.R.rand() <= 0.8

    def __call__(self, data):
        self.randomize(data)
        if self.hist_shift:
            data[self.volume_key] = self.contrast_transform(data[self.volume_key])
        if self.apply_augment:
            data[self.volume_key] = self.transform(data[self.volume_key])

        data[self.volume_key] = self.flip(data[self.volume_key])

        return data
