from mmengine.optim.scheduler import CosineAnnealingLR, LinearLR, ConstantLR

param_scheduler = [
    dict(
        type=LinearLR, start_factor=1, end_factor=1e-1, by_epoch=True, begin=0, end=500
    ),
    dict(type=CosineAnnealingLR, by_epoch=False, eta_min=0, begin=500),
]
