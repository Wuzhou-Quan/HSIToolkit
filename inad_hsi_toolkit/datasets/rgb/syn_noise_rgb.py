import mmengine, cv2, einops
from typing import Union
from ..path_provider import MultiSubsetPathProvider, MultiFilePathProvider
from ..image_provider import ImageProvider
from ..synthetic import NoiseMaker


class SyntheticNoiseRGBDataset(ImageProvider):
    def __init__(self, path_provider: Union[MultiSubsetPathProvider, MultiFilePathProvider, dict], noise_maker: Union[NoiseMaker, dict], resize_to=None, augments=None):
        if isinstance(path_provider, dict):
            path_provider = mmengine.MODELS.build(path_provider)
        self.path_manager = path_provider
        super().__init__(path_provider, augments=augments)
        if isinstance(noise_maker, dict):
            noise_maker = mmengine.MODELS.build(noise_maker)
        self.noise_maker = noise_maker
        self.resize_to = resize_to

    def __getitem__(self, index: int):
        img = super().__getitem__(index)
        if self.resize_to:
            img = cv2.resize(img, self.resize_to, interpolation=cv2.INTER_CUBIC)
        img = einops.rearrange(img, "h w c -> c h w") / 255
        img_n = self.noise_maker(img)
        img_n = img_n.clip(0, 1)
        img = img.concat([img, img_n], dim=0)
        img = self.aug(img) if self.aug else img
        return {"x": img[3:6], "gt": img[:3]}
