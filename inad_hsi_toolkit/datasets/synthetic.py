import mmengine.model
import numpy as np, mmengine


class NoiseMaker(object):
    def __init__(self, noise_bank, num_bands):
        assert len(noise_bank) == len(num_bands)
        for i in range(len(noise_bank)):
            if isinstance(noise_bank[i], dict):
                noise_bank[i] = mmengine.MODELS.build(noise_bank[i])
        self.noise_bank = noise_bank
        self.num_bands = num_bands

    def __call__(self, img: np.ndarray):
        raise NotImplementedError


class ComplexNoiseMaker(NoiseMaker):
    def __init__(self, gaussian_min, gaussian_max, noise_bank, num_bands):
        super().__init__(noise_bank, num_bands)
        assert sum(num_bands) <= 1
        self.gaussian_noise = AddGaussianNoniidBlind(gaussian_min, gaussian_max)

    def __call__(self, img: np.ndarray):
        B, _, _ = img.shape
        img = img.copy()
        img = self.gaussian_noise(img, range(B))
        all_bands = np.random.permutation(range(B))
        pos = 0
        for noise_maker, num_band in zip(self.noise_bank, self.num_bands):
            num_band = max(min(1, num_band), 0)
            num_band = int(np.floor(num_band * B))
            bands = all_bands[pos : pos + num_band]
            pos += num_band
            img = noise_maker(img, bands)
        return img


class IndependentNoiseMaker(NoiseMaker):
    """add mixed noise to the given numpy array (B,H,W)
    Args:
        noise_bank: list of noise maker (e.g. AddNoiseImpulse)
        num_bands: list of number of band which is corrupted by each item in noise_bank"""

    def __init__(self, noise_bank, num_bands):
        super().__init__(noise_bank, num_bands)

    def __call__(self, img: np.ndarray):
        B, _, _ = img.shape
        img = img.copy()
        for noise_maker, num_band in zip(self.noise_bank, self.num_bands):
            all_bands = np.random.permutation(range(B))
            num_band = max(min(1, num_band), 0)
            num_band = int(np.floor(num_band * B))
            bands = all_bands[0:num_band]
            img = noise_maker(img, bands)
        return img


class ExclusiveNoiseMaker(NoiseMaker):
    """add mixed noise to the given numpy array (B,H,W)
    Args:
        noise_bank: list of noise maker (e.g. AddNoiseImpulse)
        num_bands: list of number of band which is corrupted by each item in noise_bank"""

    def __init__(self, noise_bank, num_bands):
        super().__init__(noise_bank, num_bands)

    def __call__(self, img):
        B, H, W = img.shape
        all_bands = np.random.permutation(range(B))
        pos = 0
        for noise_maker, num_band in zip(self.noise_bank, self.num_bands):
            if 0 < num_band <= 1:
                num_band = int(np.floor(num_band * B))
            bands = all_bands[pos : pos + num_band]
            pos += num_band
            img = noise_maker(img, bands)
        return img


class AddGaussianNoniid(object):
    """add non-iid gaussian noise to the given numpy array (B,H,W)"""

    def __init__(self, sigmas):
        self.sigmas = np.array(sigmas)

    def __call__(self, img, bands):
        bwsigmas = self.sigmas[np.random.randint(0, len(self.sigmas), len(bands))]
        noise = np.random.randn(len(bands), *img.shape[1:]) * bwsigmas[:, None, None]
        img[bands] += noise
        return img


class AddGaussianNoniidBlind(object):
    """add non-iid gaussian noise to the given numpy array (B,H,W)"""

    def __init__(self, sigma_min, sigma_max):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img, bands):
        bwsigmas = np.random.uniform(self.sigma_min, self.sigma_max, len(bands))
        noise = np.random.randn(len(bands), *img.shape[1:]) * bwsigmas[:, None, None]
        img[bands] += noise
        return img


class AddStripeNoise:
    """add stripe noise to the given numpy array (B,H,W)"""

    def __init__(self, min_amount, max_amount, lam, var, direction="vertical"):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.lam = lam
        self.var = var
        self.direction = direction

    def __call__(self, img, bands):
        B, H, W = img.shape
        stype = W if self.direction == "vertical" else H
        num_stripe = np.random.randint(np.floor(self.min_amount * stype), np.floor(self.max_amount * stype), len(bands))
        for i, n in zip(bands, num_stripe):
            loc = np.random.choice(stype, n, replace=False)
            stripe = np.random.uniform(0, 1, size=(n,)) * self.lam - self.var
            if self.direction == "vertical":
                img[i, :, loc] -= np.reshape(stripe, (-1, 1))
            else:
                img[i, loc, :] -= np.reshape(stripe, (-1, 1))
        return img


class AddVerticalStripeNoise(AddStripeNoise):
    """add vertical stripe noise to the given numpy array (B,H,W)"""

    def __init__(self, min_amount, max_amount, lam, var):
        super().__init__(min_amount, max_amount, lam, var, direction="vertical")


class AddHorizontalStripeNoise(AddStripeNoise):
    """add horizontal stripe noise to the given numpy array (B,H,W)"""

    def __init__(self, min_amount, max_amount, lam, var):
        super().__init__(min_amount, max_amount, lam, var, direction="horizontal")


class AddLengthStripeNoise(object):
    """add stripe noise to the given numpy array (B,H,W)"""

    def __init__(self, min_amount, max_amount, lam, var, direction="vertical"):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.lam = lam
        self.var = var
        self.direction = direction

    def __call__(self, img, bands):
        B, H, W = img.shape
        stype = W if self.direction == "vertical" else H
        num_stripe = np.random.randint(np.floor(self.min_amount * stype), np.floor(self.max_amount * stype), len(bands))
        for i, n in zip(bands, num_stripe):
            # import ipdb
            # ipdb.set_trace()

            loc = np.random.permutation(range(stype))
            loc = loc[:n]

            for k in loc:
                if self.direction == "vertical":
                    length = np.random.randint(0, H, 1)
                    begin = np.random.randint(0, H - length + 1, 1)
                    stripe = np.random.uniform(0, 1, size=(int(length),)) * self.lam - self.var
                    img[i, int(begin) : (int(begin) + int(length)), k] -= stripe
                else:
                    length = np.random.randint(0, W, 1)
                    begin = np.random.randint(0, W - length + 1, 1)
                    stripe = np.random.uniform(0, 1, size=(int(length),)) * self.lam - self.var
                    img[i, k, int(begin) : (int(begin) + int(length))] -= stripe

        return img


class AddBroadStripeNoise(object):
    """add stripe noise to the given numpy array (B,H,W)"""

    def __init__(self, min_amount, max_amount, lam, var, direction="vertical"):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.lam = lam
        self.var = var
        self.direction = direction

    def __call__(self, img, bands):
        B, H, W = img.shape
        stype = W if self.direction == "vertical" else H
        num_stripe = np.random.randint(np.floor(self.min_amount * stype), np.floor(self.max_amount * stype), len(bands))
        for i, n in zip(bands, num_stripe):
            perio = 10
            loc = np.random.permutation(range(0, (stype - perio), perio))
            loc = loc[: int(n // perio)]
            for k in loc:
                stripe = np.random.uniform(0, 1, size=(perio,)) * self.lam - self.var
                if self.direction == "vertical":
                    img[i, :, k : (k + int(perio))] -= np.tile(stripe, (H, 1))

                else:
                    img[i, k : (k + int(perio)), :] -= np.transpose(np.tile(stripe, (W, 1)), (1, 0))

        return img


class AddPerioStripeNoise(object):
    """add stripe noise to the given numpy array (B,H,W)"""

    def __init__(self, lam, var):
        self.lam = lam
        self.var = var

    def __call__(self, img, bands):
        B, H, W = img.shape
        perio = 80
        num_stripe = int(W // perio)
        stripe = np.random.uniform(0, 1, size=(perio - 30,)) * self.lam - self.var
        for i in bands:
            for k in range(num_stripe):
                img[i, k * int(perio) + 16 : (k + 1) * int(perio) - 30 + 16, :] -= np.transpose(np.tile(stripe, (W, 1)), (1, 0))
        return img


class AddNoiseImpulse(object):
    """add impulse noise to the given numpy array (B,H,W)"""

    def __init__(self, amounts, s_vs_p=0.5):
        self.amounts = np.array(amounts)
        self.s_vs_p = s_vs_p

    def __call__(self, img, bands):
        # bands = np.random.permutation(range(img.shape[0]))[:self.num_band]
        bwamounts = self.amounts[np.random.randint(0, len(self.amounts), len(bands))]
        for i, amount in zip(bands, bwamounts):
            self.add_noise(img[i, ...], amount=amount, salt_vs_pepper=self.s_vs_p)
        return img

    def add_noise(self, image, amount, salt_vs_pepper):
        # out = image.copy()
        out = image
        p = amount
        q = salt_vs_pepper
        flipped = np.random.choice([True, False], size=image.shape, p=[p, 1 - p])
        salted = np.random.choice([True, False], size=image.shape, p=[q, 1 - q])
        peppered = ~salted
        out[flipped & salted] = 1
        out[flipped & peppered] = 0
        return out


class AddNoiseDeadline(object):
    """add deadline noise to the given numpy array (B,H,W)"""

    def __init__(self, min_amount, max_amount):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount

    def __call__(self, img, bands):
        B, H, W = img.shape
        # bands = np.random.permutation(range(img.shape[0]))[:len(bands)]
        num_deadline = np.random.randint(np.ceil(self.min_amount * W), np.ceil(self.max_amount * W), len(bands))
        for i, n in zip(bands, num_deadline):
            loc = np.random.permutation(range(W))
            loc = loc[:n]
            img[i, :, loc] = 0
        return img


class AddSuperMixedNoiseH(IndependentNoiseMaker):
    def __init__(self):
        self.noise_bank = [
            AddGaussianNoniid([0.1, 0.2, 0.3, 0.4]),
            AddStripeNoise(0.05, 0.25, 0.7, 0.25),
            AddBroadStripeNoise(0.05, 0.45, 0.7, 0.25),
            AddLengthStripeNoise(0.05, 0.45, 0.7, 0.25),
        ]
        self.num_bands = [1, 1, 1]


class AddMixedNoiseH(IndependentNoiseMaker):
    def __init__(self):
        self.noise_bank = [AddGaussianNoniid([0.1, 0.2, 0.3, 0.4]), AddVerticalStripeNoise(0.05, 0.25, 1, 0.25)]
        self.num_bands = [0, 1]


class AddMixedNoiseW(IndependentNoiseMaker):
    def __init__(self):
        self.noise_bank = [AddGaussianNoniid([0.1, 0.2, 0.3, 0.4]), AddHorizontalStripeNoise(0.05, 0.25, 1, 0.25)]
        self.num_bands = [1, 1]


class AddStripeNoiseH(ExclusiveNoiseMaker):
    def __init__(self):
        self.noise_bank = [AddGaussianNoniid([0]), AddVerticalStripeNoise(0.05, 0.45, 0.7, 0.5)]
        self.num_bands = [0, 2 / 3]


class AddStripeNoiseW(ExclusiveNoiseMaker):
    def __init__(self):
        self.noise_bank = [AddGaussianNoniid([0]), AddHorizontalStripeNoise(0.05, 0.45, 0.7, 0.5)]
        self.num_bands = [0, 2 / 3]


class AddStripeNoiseL(ExclusiveNoiseMaker):
    def __init__(self):
        self.noise_bank = [AddGaussianNoniid([0]), AddLengthStripeNoise(0.05, 0.45, 0.7, 0.5)]
        self.num_bands = [0, 2 / 3]


class AddStripeNoiseB(ExclusiveNoiseMaker):
    def __init__(self):
        self.noise_bank = [AddGaussianNoniid([0]), AddBroadStripeNoise(0.02, 0.15, 0.7, 0.5)]
        self.num_bands = [0, 2 / 3]


class AddStripeNoiseP(ExclusiveNoiseMaker):
    def __init__(self):
        self.noise_bank = [AddGaussianNoniid([0]), AddPerioStripeNoise(0.7, 0.3)]
        self.num_bands = [0, 2 / 3]


class AddNoiidNoise(ExclusiveNoiseMaker):
    def __init__(self):
        self.noise_bank = [AddGaussianNoniid([0]), AddGaussianNoniid([50])]
        self.num_bands = [0, 2 / 3]


class AddNoiidBlind(ExclusiveNoiseMaker):
    def __init__(self):
        self.noise_bank = [AddGaussianNoniid([0]), AddGaussianNoniid([30, 50, 70])]
        self.num_bands = [0, 2 / 3]


class AddErosion(ExclusiveNoiseMaker):
    def __init__(self):
        self.noise_bank = [AddGaussianNoniid([0]), AddNoiseImpulse([0.05, 0.1, 0.15], 0.5)]
        self.num_bands = [0, 2 / 3]


class AddResolutionZoomBlurring(ExclusiveNoiseMaker):
    def __init__(self):
        self.noise_bank = [AddGaussianNoniid([0]), AddNoiseImpulse([0.05, 0.1, 0.15], 0.5)]
        self.num_bands = [0, 2 / 3]
