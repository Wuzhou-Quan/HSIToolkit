import pathlib, spectral.io.envi as envi, pathlib


class HDRSingle2PatchesDataset:
    def __init__(self, path: pathlib.Path, color_bit=8, patch_size=(256, 256), save_last=True, augments=None):
        hsi = envi.open(path)
        hsi = hsi[:, :, :]
        tH, tW, C = hsi.shape
        patch_h, patch_w = tH // patch_size[0], tW // patch_size[1]
        print(f"Ver: {patch_h+1 if save_last else 0} Hori: {patch_w+1 if save_last else 0}")

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

        self.color_bit = color_bit
        self.aug = augments

    def get_item_without_aug(self, index: int):
        patch = self.patches[index] / (2**self.color_bit - 1)
        return {"x": patch, "gt": patch}

    def __getitem__(self, index: int):
        patch = self.patches[index] / (2**self.color_bit - 1)
        patch = self.aug(patch) if self.aug else patch
        return {"x": patch, "gt": patch}

    def __len__(self):
        return len(self.patches)
