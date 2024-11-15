import click, scipy.io as io, numpy, pathlib


@click.command()
@click.option("--source_folder", type=str)
@click.option("--out_folder", type=str)
@click.option("--res", type=int, default=64)
def slice(source_folder, out_folder, res):
    source_folder = pathlib.Path(source_folder)
    out_folder = pathlib.Path(out_folder)
    assert source_folder.is_dir()
    assert not out_folder.is_file()
    if not out_folder.exists():
        out_folder.mkdir(parents=True)
    for idx, p in enumerate(source_folder.iterdir()):
        data = io.loadmat(p)
        for i in range(512 // res):
            for j in range(512 // res):
                gt = data["gt"][i * res : (i + 1) * res, j * res : (j + 1) * res]
                numpy.save(out_folder / f"{p.stem}_{i}_{j}.npy", gt)


if __name__ == "__main__":
    slice()
