from inad_hsi_toolkit.datasets.synthetic import IndependentNoiseMaker, AddGaussianNoniid, AddStripeNoise, AddNoiseImpulse, AddNoiseDeadline

strip_noise_direction = "vertical"
noise_maker = dict(
    type=IndependentNoiseMaker,
    noise_bank=[
        dict(type=AddGaussianNoniid, sigmas=[10 / 255, 30 / 255, 50 / 255, 70 / 255]),
        dict(type=AddStripeNoise, min_amount=0.05, max_amount=0.15, lam=0.5, var=0.25, direction=strip_noise_direction),
        dict(type=AddNoiseImpulse, amounts=[0.1, 0.3, 0.5, 0.7]),
        dict(type=AddNoiseDeadline, min_amount=0.05, max_amount=0.15),
    ],
    num_bands=[1, 1 / 3, 1 / 3, 1 / 3],
)
