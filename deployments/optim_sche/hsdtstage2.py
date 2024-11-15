from mmengine.optim.scheduler import ConstantLR
import math

param_scheduler = [
    dict(type=ConstantLR, factor=0.1, begin=15, end=26),
    dict(type=ConstantLR, factor=0.05, begin=25, end=31),
    dict(type=ConstantLR, factor=0.01, begin=30, end=36),
    dict(type=ConstantLR, factor=0.005, begin=35, end=46),
    dict(type=ConstantLR, factor=0.001, begin=45, end=math.inf),
]
