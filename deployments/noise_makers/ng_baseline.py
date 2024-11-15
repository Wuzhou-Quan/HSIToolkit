from inad_hsi_toolkit.datasets.synthetic import IndependentNoiseMaker, AddGaussianNoniid

strip_noise_direction = "vertical"
noise_maker = dict(
    type=IndependentNoiseMaker,
    noise_bank=[
        dict(type=AddGaussianNoniid, sigmas=[0.1, 0.2, 0.3, 0.4]),
    ],
    num_bands=[1],
)
