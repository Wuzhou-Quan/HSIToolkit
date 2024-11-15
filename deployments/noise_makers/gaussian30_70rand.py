from inad_hsi_toolkit.datasets.synthetic import IndependentNoiseMaker, AddGaussianNoniidBlind

strip_noise_direction = "vertical"
noise_maker = dict(
    type=IndependentNoiseMaker,
    noise_bank=[
        dict(type=AddGaussianNoniidBlind, sigma_min=30 / 255, sigma_max=70 / 255),
    ],
    num_bands=[1],
)
