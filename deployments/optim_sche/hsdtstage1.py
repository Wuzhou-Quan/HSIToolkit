from mmengine.optim.scheduler import ConstantLR
import math

param_scheduler = [dict(type=ConstantLR, factor=0.1, begin=20, end=math.inf)]
