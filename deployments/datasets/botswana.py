from inad_hsi_toolkit.datasets.patches_dataset.syn_single_patched_dataset import SyntheticSingle2PatchesDataset
import gsettings


color_bit = 16
channels = 145
path = gsettings.datasets_folder_path / "Botswana" / "Botswana.npy"

dataset = dict(
    type=SyntheticSingle2PatchesDataset,
    path=path,
    color_bit=color_bit,
    save_last=True,
    pca_out_channels=None,
    train_set_ratio=0.9,
)