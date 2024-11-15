import mmengine, cv2, einops, torch, numpy
from typing import Union
from .path_provider import MultiSubsetPathProvider, MultiFilePathProvider
from .image_provider import ImageProvider


class PreAugDataset(ImageProvider):
    def __init__(self, path_provider: Union[MultiSubsetPathProvider, MultiFilePathProvider, dict]):
        if isinstance(path_provider, dict):
            path_provider = mmengine.MODELS.build(path_provider)
        self.path_manager = path_provider
        super().__init__(path_provider)

    def __getitem__(self, index: int):
        img_pair = super().__getitem__(index)
        # x = img_pair["x"]
        # x2 = img_pair["gt"]
        return img_pair