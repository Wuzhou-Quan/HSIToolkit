import pathlib, numpy, scipy, h5py, cv2, torch


def load_mat(file_path: pathlib.Path) -> dict:
    data = None
    try:
        data = scipy.io.loadmat(file_path)
    except:
        data_tmp = h5py.File(file_path, "r")
        data = {}
        for key in data_tmp:
            data[key] = numpy.array(data_tmp[key])
    return data


def load_tif(path: pathlib.Path):
    import libtiff

    libtiff.libtiff_ctypes.suppress_warnings()
    tiff: libtiff.TIFF = libtiff.TIFF.open(path)
    tiff = tiff.read_image()
    return numpy.array(tiff)


def load_bit(path: pathlib.Path):
    ret = cv2.imread(str(path))
    return ret


def load_numpy(path: pathlib.Path):
    ret = numpy.load(path)
    return ret


def load_torchtensor(path: pathlib.Path):
    ret = torch.load(path)
    return ret


file_loader = {
    ".tif": load_tif,
    ".tiff": load_tif,
    ".mat": load_mat,
    ".jpeg": load_bit,
    ".npy": load_numpy,
    ".pt": load_torchtensor,
}
