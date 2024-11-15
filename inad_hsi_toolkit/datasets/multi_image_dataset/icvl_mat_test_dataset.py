import numpy, random, torch, pathlib, torchvision.transforms as v2, mmengine
from typing import Union
from ..synthetic import NoiseMaker
from ..file_loader import file_loader


class ICVLTestDataset:
    def __init__(self, root_path: pathlib.Path, noise_maker: Union[NoiseMaker, dict] = None, color_bit=8, augments=None, valid_suffix=[".mat"]):
        self.color_bit = color_bit
        self.image_path_list = []
        for image_path in root_path.glob("*"):
            if image_path.suffix.lower() in valid_suffix:
                self.image_path_list.append(image_path)
        self.noise_maker = mmengine.MODELS.build(noise_maker) if noise_maker is not None else None
        self.aug = augments

    def getitem(self, index: int, enable_aug: bool = False, enable_noise: bool = False):
        x = file_loader[self.image_path_list[index].suffix.lower()](self.image_path_list[index])
        gt = x["gt"]
        x = x["input"] if enable_noise else gt
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
