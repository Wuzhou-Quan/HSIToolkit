from inad_hsi_toolkit.datasets.multi_image_dataset.icvl_mat_test_dataset import ICVLTestDataset
import gsettings

channels = 31

dataset = dict(
    type=ICVLTestDataset,
    root_path=gsettings.datasets_folder_path / "icvl_test" / "icvl_512_30",
    color_bit=0,
)
