from inad_hsi_toolkit.datasets.multi_image_dataset.split_file_dataset import SplitFileDataset
import gsettings, pathlib

channels = 202

dataset = dict(
    type=SplitFileDataset,
    root_path=gsettings.datasets_folder_path / "hyspecnet11k" / "patches",
    split_file=[gsettings.datasets_folder_path / "hyspecnet11k" / "splits" / "easy" / "train.csv"],
    color_bit=0,
)
val_dataset = dict(
    type=SplitFileDataset,
    root_path=gsettings.datasets_folder_path / "hyspecnet11k" / "patches",
    split_file=[gsettings.datasets_folder_path / "hyspecnet11k" / "splits" / "easy" / "val.csv", gsettings.datasets_folder_path / "hyspecnet11k" / "splits" / "easy" / "test.csv"],
    color_bit=0,
)
