from inad_hsi_toolkit.datasets.synthetic import IndependentNoiseMaker, AddGaussianNoniid, AddStripeNoise, AddBroadStripeNoise, AddLengthStripeNoise

strip_noise_direction = "vertical"
noise_maker = dict(
    type=IndependentNoiseMaker,
    noise_bank=[
        dict(type=AddGaussianNoniid, sigmas=[0.1, 0.2, 0.3, 0.4]),
        dict(type=AddStripeNoise, min_amount=0.05, max_amount=0.25, lam=0.7, var=0.25, direction=strip_noise_direction),
    ],
    num_bands=[1, 1],
)
