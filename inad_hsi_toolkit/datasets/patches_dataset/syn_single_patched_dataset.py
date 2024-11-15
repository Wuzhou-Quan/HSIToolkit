import numpy, mmengine, einops, random, torch
# from models.hsimae.Utils.group_pca import group_pca
from typing import Union
from ..synthetic import NoiseMaker
from ..file_loader import file_loader
from .patches_dataset import PatchesDataset


class SyntheticSingle2PatchesDataset:
    def __init__(self, path, noise_maker: Union[NoiseMaker, dict], color_bit=8, patch_size=(256, 256), pca_out_channels=None, save_last=True, augments=None, train_set_ratio=1):
        hsi = file_loader[path.suffix.lower()](path)  # H W C
        self.color_bit = color_bit
        assert hsi.min() >= 0
        assert hsi.max() <= (2**self.color_bit - 1)
        tH, tW, C = hsi.shape
        self.color_bit = color_bit
        patch_h, patch_w = tH // patch_size[0], tW // patch_size[1]
        hsi = hsi / (2**self.color_bit - 1)

        self.patches = []
        for i in range(patch_h):
            for j in range(patch_w):
                patch = hsi[i * patch_size[0] : (i + 1) * patch_size[0], j * patch_size[1] : (j + 1) * patch_size[1], :]
                self.patches.append(patch)
        if save_last:
            for j in range(patch_w):
                patch = hsi[tH - patch_size[0] :, j * patch_size[1] : (j + 1) * patch_size[1], :]
                self.patches.append(patch)
            for i in range(patch_h):
                patch = hsi[i * patch_size[0] : (i + 1) * patch_size[0], tW - patch_size[1] :, :]
                self.patches.append(patch)

        total_patches = len(self.patches)
        test_set_size = int(total_patches * (1 - train_set_ratio))
        interval = total_patches // test_set_size

        test_indices = list(range(0, total_patches, interval))[:test_set_size]

        self.test_patches = []
        self.train_patches = []
        for i in range(total_patches):
            if i in test_indices:
                self.test_patches.append(self.patches[i])
            else:
                self.train_patches.append(self.patches[i])

        self.patches = PatchesDataset(self.train_patches, noise_maker, augments)
        self.noise_maker = noise_maker
        self.augments = augments
        self.src_image = hsi

    def get_test_set(self) -> PatchesDataset:
        return PatchesDataset(self.test_patches, self.noise_maker, self.augments)

    def getitem(self, index: int, enable_aug: bool = False, enable_noise: bool = False):
        return self.patches.getitem(index, enable_aug, enable_noise)

    def __getitem__(self, index: int):
        """
        NOTE!: Only for training dataset!
        """
        return self.getitem(index, True, True)

    def __len__(self):
        return len(self.patches)
