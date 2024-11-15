from mmengine.model import BaseModel
import mmengine, torch


def l1_regularization(model):
    l1_loss = torch.tensor(0.0, requires_grad=True)
    for param in model.parameters():
        l1_loss = l1_loss + torch.sum(torch.abs(param))
    return l1_loss


class TrainingWrapper(BaseModel):
    def __init__(self, model, loss):
        super().__init__()
        self.model = mmengine.MODELS.build(model)
        self.loss = mmengine.MODELS.build(loss)

    def forward(self, x, gt=None, mode="predict"):
        y = self.model(x)
        if mode == "loss":
            l1loss = self.loss(y, gt) * 10000
            return {"loss": l1loss}
        elif mode == "predict":
            return y


class TrainingWrapperWithSVDLoss(BaseModel):
    def __init__(self, model, loss):
        super().__init__()
        self.model = mmengine.MODELS.build(model)
        self.loss = mmengine.MODELS.build(loss)

    def forward(self, x, gt=None, mode="predict"):
        y = self.model(x)
        if mode == "loss":
            l1loss = self.loss(y, gt) * 10000
            return {"loss": l1loss}
        elif mode == "predict":
            return y
