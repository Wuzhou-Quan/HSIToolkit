from inad_hsi_toolkit.datasets.multi_image_dataset.split_file_dataset import SplitFileDataset
import gsettings

channels = 31

dataset = dict(
    type=SplitFileDataset,
    root_path=gsettings.datasets_folder_path / "icvl" / "imgs",
    split_file=[gsettings.datasets_folder_path / "icvl" / "splits" / "group1.txt"],
    color_bit=12,
)
