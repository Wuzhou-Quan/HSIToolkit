import os, pathlib

interval = 8
path = pathlib.Path("/home/walterd/projects/mambahsi/data/synthetic_data/xiongan")


def divide_train_val(path: pathlib.Path, interval: int):
    train_path = path / "train"
    val_path = path / "val"
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    for i, file in enumerate(path.iterdir()):
        if file.is_dir():
            continue
        if i % interval == 0:
            os.rename(file, val_path / file.name)
        else:
            os.rename(file, train_path / file.name)


if __name__ == "__main__":
    divide_train_val(path, interval)
