import mmengine, cv2
from typing import Union
from ..path_provider import MultiSubsetPathProvider, MultiFilePathProvider
from ..image_provider import ImageProvider


class VenillaHSIDataset(ImageProvider):
    def __init__(self, path_provider: Union[MultiSubsetPathProvider, MultiFilePathProvider, dict], color_bit=8, resize_to=None, augments=None):
        if isinstance(path_provider, dict):
            path_provider = mmengine.MODELS.build(path_provider)
        self.path_manager = path_provider
        super().__init__(path_provider, augments=augments)
        self.resize_to = resize_to
        self.color_bit = color_bit

    def __getitem__(self, index: int):
        img = super().__getitem__(index) / (2**self.color_bit - 1)
        if self.resize_to:
            img = cv2.resize(img, self.resize_to, interpolation=cv2.INTER_CUBIC)
        img = self.aug(img) if self.aug else img
        return {"x": img, "gt": img}
