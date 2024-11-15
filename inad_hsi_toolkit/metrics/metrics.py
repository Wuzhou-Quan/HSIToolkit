import numpy, torch, math
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim


def PSNR(a, b, bit_depth=0):
    a, b = map(lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x, [a, b])
    mse = numpy.mean((a - b) ** 2)
    return math.inf if mse == 0 else 20 * numpy.log10((2**bit_depth) / numpy.sqrt(mse))


def MPSNR(a, b, bit_depth=0):
    a, b = map(lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x, [a, b])
    mse = numpy.mean((a - b) ** 2, axis=(1, 2))
    return math.inf if numpy.all(mse == 0) else 20 * numpy.log10((2**bit_depth) / numpy.sqrt(mse))


def SSIM(a, b):
    a, b = map(lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x, [a, b])
    return ssim(a, b, multichannel=True, data_range=1.0)


def MSSIM(a, b):
    a, b = map(lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x, [a, b])
    return ssim(a, b, multichannel=True, data_range=1.0, full=True)[0]


def MAE(a, b):
    a, b = map(lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x, [a, b])
    return numpy.mean(numpy.abs(a - b))


def MSE(a, b):
    a, b = map(lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x, [a, b])
    return numpy.mean((a - b) ** 2)


def SAM(pred, gt):
    pred, gt = map(lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x, [pred, gt])
    assert pred.ndim == 3 and pred.shape == gt.shape
    sam_rad = numpy.zeros([pred.shape[0], pred.shape[1]])
    for x in range(pred.shape[0]):
        for y in range(pred.shape[1]):
            tmp_pred = pred[x, y].ravel()
            tmp_true = gt[x, y].ravel()
            cos_theta = numpy.dot(tmp_pred, tmp_true) / (numpy.linalg.norm(tmp_pred) * numpy.linalg.norm(tmp_true))
            sam_rad[x, y] = numpy.arccos(cos_theta)
    sam_deg = sam_rad.mean() * 180 / numpy.pi
    return sam_deg
