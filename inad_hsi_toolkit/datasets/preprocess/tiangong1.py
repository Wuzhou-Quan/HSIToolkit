import pathlib, re, libtiff, numpy
libtiff.libtiff_ctypes.suppress_warnings()
if __name__ == "__main__":
    root_path = pathlib.Path("data/tiangong1").absolute()
    output_path = pathlib.Path("data/tiangong1_preprocessed").absolute()
    if not output_path.is_dir():
        output_path.mkdir()
    assert root_path.is_dir()
    for subset in root_path.iterdir():
        if not subset.is_dir():
            continue
        img_pairs = {}
        for file in subset.iterdir():
            name = file.stem
            pat = re.search(r"[0-9]{3}_([A-Z]{3})_([0-9]{10})", name)
            wv = pat[1]
            time = pat[2]
            if time not in img_pairs:
                img_pairs[time] = {}
            array = libtiff.TIFF.open(file).read_image()
            img_pairs[time][wv] = array
        for time in img_pairs:
            print(f"{subset.name}_{time}: {len(img_pairs[time])}")
            numpy.save(output_path / f"{subset.name}_{time}", img_pairs[time])
