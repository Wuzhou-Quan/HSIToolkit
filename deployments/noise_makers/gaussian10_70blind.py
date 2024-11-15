from inad_hsi_toolkit.datasets.synthetic import IndependentNoiseMaker, AddGaussianNoniid

strip_noise_direction = "vertical"
noise_maker = dict(
    type=IndependentNoiseMaker,
    noise_bank=[
        dict(type=AddGaussianNoniid, sigmas=[70 / 255, 50 / 255, 30 / 255, 10 / 255]),
    ],
    num_bands=[1],
)
