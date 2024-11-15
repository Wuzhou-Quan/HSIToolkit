from inad_hsi_toolkit.datasets.synthetic import IndependentNoiseMaker, AddGaussianNoniid, AddStripeNoise, AddBroadStripeNoise, AddLengthStripeNoise

strip_noise_direction = "vertical"
noise_maker = dict(
    type=IndependentNoiseMaker,
    noise_bank=[
        dict(type=AddGaussianNoniid, sigmas=[30 / 255]),
    ],
    num_bands=[1],
)
