import numpy, mmengine, random, torch, torchvision.transforms as v2
from typing import Union
from ..synthetic import NoiseMaker


class PatchesDataset:
    def __init__(self, patches: list, noise_maker: Union[NoiseMaker, dict], augments=None):
        self.noise_maker = mmengine.MODELS.build(noise_maker) if noise_maker is not None else None
        self.patches = patches
        self.aug = augments

    def getitem(self, index: int, enable_aug: bool = False, enable_noise: bool = False):
        gt = self.patches[index]
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
        return self.getitem(index, enable_noise=True)

    def __len__(self):
        return len(self.patches)
