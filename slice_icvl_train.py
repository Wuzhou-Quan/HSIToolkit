import click, pathlib, numpy as np, einops
from inad_hsi_toolkit.datasets.file_loader import file_loader
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Manager
from rich.progress import Progress, BarColumn, TextColumn


class PatchSlicer:
    def __init__(self, source_folder, output_folder, color_bit_depth, patch_size):
        self.output_folder = output_folder
        self.color_bit_depth = color_bit_depth
        self.patch_size = patch_size
        self.files = list(source_folder.iterdir())
        self.total_patches = ((1392 // patch_size[0]) + 1) * ((1300 // patch_size[1]) + 1) * len(self.files)

    def save_patch(self, file_stem, i, j, patch):
        np.save(self.output_folder / f"{file_stem}_{i}_{j}.npy", patch)

    def process_file(self, file):
        hsi_chw = file_loader[file.suffix.lower()](file)["rad"]
        hsi_hwc = einops.rearrange(hsi_chw, "c h w -> h w c")
        assert hsi_hwc.min() >= 0
        assert hsi_hwc.max() <= (2**self.color_bit_depth - 1)
        tH, tW, C = hsi_hwc.shape
        patch_h, patch_w = tH // self.patch_size[0], tW // self.patch_size[1]

        patches = []
        for i in range(patch_h):
            for j in range(patch_w):
                patch = hsi_hwc[i * self.patch_size[0] : (i + 1) * self.patch_size[0], j * self.patch_size[1] : (j + 1) * self.patch_size[1], :]
                patches.append((file.stem, i, j, patch))
        for j in range(patch_w):
            patch = hsi_hwc[tH - self.patch_size[0] :, j * self.patch_size[1] : (j + 1) * self.patch_size[1], :]
            patches.append((file.stem, patch_h, j, patch))
        for i in range(patch_h):
            patch = hsi_hwc[i * self.patch_size[0] : (i + 1) * self.patch_size[0], tW - self.patch_size[1] :, :]
            patches.append((file.stem, i, patch_w, patch))

        with ThreadPoolExecutor() as thread_executor:
            save_futures = [thread_executor.submit(self.save_patch, file_stem, i, j, patch) for file_stem, i, j, patch in patches]
            for save_future in as_completed(save_futures):
                try:
                    save_future.result()
                except Exception as e:
                    print(f"Exception occurred while saving: {e}")
                self.progress_dict[file.name] += 1

    def start_progress(self):
        manager = Manager()
        progress_dict = self.progress_dict = manager.dict({file.name: 0 for file in self.files})

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TextColumn("{task.completed}/{task.total} patches"),
        ) as main_progress:
            main_task_id = main_progress.add_task("Processing files...", total=self.total_patches)
            with ProcessPoolExecutor() as process_executor:
                futures = [process_executor.submit(self.process_file, file) for file in self.files]
                while any(future.running() for future in futures):
                    completed_patches = sum(progress_dict.values())
                    main_progress.update(main_task_id, completed=completed_patches)

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Exception occurred: {e}")
            main_progress.remove_task(main_task_id)


@click.command()
@click.option("--source_folder", type=str)
@click.option("--out_folder", type=str)
@click.option("--res", type=int, default=64)
@click.option("--color_bit_depth", type=int, default=0)
def slice_icvl_train(res, source_folder, out_folder, color_bit_depth):
    assert pathlib.Path(source_folder).is_dir()
    assert not pathlib.Path(out_folder).is_file()
    if not pathlib.Path(out_folder).exists():
        pathlib.Path(out_folder).mkdir(parents=True, exist_ok=False)
    slicer = PatchSlicer(pathlib.Path(source_folder), pathlib.Path(out_folder), color_bit_depth, (res, res))
    slicer.start_progress()


if __name__ == "__main__":
    slice_icvl_train()
