import pathlib
import numpy as np
import einops, gsettings
from inad_hsi_toolkit.datasets.file_loader import file_loader
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Manager
from rich.progress import Progress, BarColumn, TextColumn

# 配置参数
color_bit_depth = 12
patch_size = [128, 128]
output_folder = pathlib.Path(gsettings.datasets_folder_path / "icvl" / "imgs")
spectral_file_folder = pathlib.Path(gsettings.datasets_folder_path / "ICVL")


# 定义保存 patch 的函数
def save_patch(file_stem, i, j, patch):
    np.save(output_folder / f"{file_stem}_{i}_{j}.npy", patch)


# 定义处理每个文件的函数
def process_file(file, progress_dict):
    hsi_chw = file_loader[file.suffix.lower()](file)["rad"]
    hsi_hwc = einops.rearrange(hsi_chw, "c h w -> h w c")
    assert hsi_hwc.min() >= 0
    assert hsi_hwc.max() <= (2**color_bit_depth - 1)
    tH, tW, C = hsi_hwc.shape
    patch_h, patch_w = tH // patch_size[0], tW // patch_size[1]

    patches = []
    for i in range(patch_h):
        for j in range(patch_w):
            patch = hsi_hwc[i * patch_size[0] : (i + 1) * patch_size[0], j * patch_size[1] : (j + 1) * patch_size[1], :]
            patches.append((file.stem, i, j, patch))
    for j in range(patch_w):
        patch = hsi_hwc[tH - patch_size[0] :, j * patch_size[1] : (j + 1) * patch_size[1], :]
        patches.append((file.stem, patch_h, j, patch))
    for i in range(patch_h):
        patch = hsi_hwc[i * patch_size[0] : (i + 1) * patch_size[0], tW - patch_size[1] :, :]
        patches.append((file.stem, i, patch_w, patch))

    with ThreadPoolExecutor() as thread_executor:
        save_futures = [thread_executor.submit(save_patch, file_stem, i, j, patch) for file_stem, i, j, patch in patches]
        for save_future in as_completed(save_futures):
            try:
                save_future.result()
            except Exception as e:
                print(f"Exception occurred while saving: {e}")
            # 更新进度字典
            progress_dict[file.name] += 1


# 使用多进程处理所有文件，并显示总进度条
files = list(spectral_file_folder.iterdir())
total_patches = ((1392 // 128) + 1) * ((1300 // 128) + 1) * 100

manager = Manager()
progress_dict = manager.dict({file.name: 0 for file in files})

with Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
    TextColumn("{task.completed}/{task.total} patches"),
) as main_progress:
    main_task_id = main_progress.add_task("Processing files...", total=total_patches)
    with ProcessPoolExecutor() as process_executor:
        futures = [process_executor.submit(process_file, file, progress_dict) for file in files]
        while any(future.running() for future in futures):
            completed_patches = sum(progress_dict.values())
            main_progress.update(main_task_id, completed=completed_patches)

    for future in as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"Exception occurred: {e}")
    main_progress.remove_task(main_task_id)
