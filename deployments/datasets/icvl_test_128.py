from inad_hsi_toolkit.datasets.multi_image_dataset.folder_dataset import FolderDataset
import gsettings

channels = 31
dataset = dict(
    type=FolderDataset,
    root_path=gsettings.datasets_folder_path / "icvl" / "test_imgs_128",
    color_bit=0,
)
