import numpy, random, torch, pathlib, torchvision.transforms as v2, mmengine
from typing import Union
from ..synthetic import NoiseMaker
from ..file_loader import file_loader


class SplitFileDataset:
    def __init__(self, root_path: pathlib.Path, split_file: pathlib.Path, noise_maker: Union[NoiseMaker, dict] = None, color_bit=8, augments=None, valid_suffix=[".tif", ".npy"]):
        self.color_bit = color_bit
        self.image_path_list = []
        if isinstance(split_file, str):
            with open(split_file, "r") as f:
                for image_path in f.readlines():
                    path = root_path / image_path.strip()
                    self.image_path_list.append(path)
        elif isinstance(split_file, list):
            for f in split_file:
                with open(f, "r") as f:
                    for image_path in f.readlines():
                        path = root_path / image_path.strip()
                        self.image_path_list.append(path)
        self.noise_maker = mmengine.MODELS.build(noise_maker) if noise_maker is not None else None
        self.aug = augments

    def getitem(self, index: int, enable_aug: bool = False, enable_noise: bool = False):
        gt = file_loader[self.image_path_list[index].suffix.lower()](self.image_path_list[index])

        gt = gt / 2**self.color_bit
        x = self.noise_maker(gt) if enable_noise and self.noise_maker is not None else gt
        if self.aug and enable_aug:
            # Save the random state
            state = random.getstate()
            np_state = numpy.random.get_state()
            torch_state = torch.get_rng_state()

            # Apply augmentation to gt
            gt = self.aug(gt)

            # Restore the random state
            random.setstate(state)
            numpy.random.set_state(np_state)
            torch.set_rng_state(torch_state)

            # Apply the same augmentation to x
            x = self.aug(x)
        else:
            trans = v2.Compose(
                [
                    v2.ToTensor(),
                    v2.ConvertImageDtype(torch.float32),
                ]
            )
            x = trans(x)
            gt = trans(gt)
        return {"x": x, "gt": gt}

    def __getitem__(self, index: int):
        """
        NOTE!: Only for training dataset!
        """
        return self.getitem(index, True, True)

    def __len__(self):
        return len(self.image_path_list)
