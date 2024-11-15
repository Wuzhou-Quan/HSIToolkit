from inad_hsi_toolkit.datasets.synthetic import IndependentNoiseMaker, AddStripeNoise

strip_noise_direction = "vertical"
noise_maker = dict(
    type=IndependentNoiseMaker,
    noise_bank=[
        dict(type=AddStripeNoise, min_amount=0.05, max_amount=0.25, lam=0.7, var=0.25, direction=strip_noise_direction),
    ],
    num_bands=[1],
)
