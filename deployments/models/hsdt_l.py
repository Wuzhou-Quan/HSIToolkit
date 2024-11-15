import torch
from models.hsdt.arch import HSDT
from inad_hsi_toolkit.miscs.mme_general_trainer_wrapper import TrainingWrapper

net = dict(
    type=HSDT,
    in_channels=1,
    channels=16,
    num_half_layer=7,
    sample_idx=[1, 3, 5],
    Fusion=None,
)
loss = dict(type=torch.nn.MSELoss)
model = dict(type=TrainingWrapper, model=net, loss=loss)
