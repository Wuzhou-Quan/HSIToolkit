# %%
import gsettings

split_file_folder = gsettings.datasets_folder_path / "hyspecnet11k" / "splits" / "easy"
splits = {"train": split_file_folder / "train.csv", "test": split_file_folder / "test.csv", "val": split_file_folder / "val.csv"}

# 清洗成为Npy
clean_folder = gsettings.datasets_folder_path / "hyspecnet11k" / "clean_patches"
for set in splits:
    with open(splits[set], "r") as f:
        files = f.readlines()
    for file in files:
        file = gsettings.datasets_folder_path / "hyspecnet11k" / "patches" / file.strip()
        
    with open(gsettings.datasets_folder_path / "hyspecnet11k" / "splits" / f"{set}.csv", "w+") as f:
        f.writelines(lines)

# 将csv中逐行的-DATA.npy尾缀全部更换为-SPECTRAL_IMAGE.TIF
# for set in splits:
#     with open(splits[set], "r") as f:
#         lines = f.readlines()
#         lines = [line.replace("-DATA.npy", "-SPECTRAL_IMAGE.TIF") for line in lines]
#     print(gsettings.datasets_folder_path / "hyspecnet11k" / "splits" / f"{set}.csv")
#     with open(gsettings.datasets_folder_path / "hyspecnet11k" / "splits" / f"{set}.csv", "w+") as f:
#         f.writelines(lines)
