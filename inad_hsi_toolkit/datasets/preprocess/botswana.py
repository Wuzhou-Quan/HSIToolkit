# %%
import pathlib, numpy, gsettings
from inad_datasets.file_loader import file_loader

# %% Handle Pavia
# |- $gsettings.datasets_folder_path$
# |-- Botswana
# |--- Botswana.mat
# |--+ [CREATED]Botswana.npy
spectral_file = pathlib.Path(gsettings.datasets_folder_path / "Botswana" / "Botswana.mat")
hsi = file_loader[spectral_file.suffix.lower()](spectral_file)
print(hsi.keys())  # ['__header__', '__version__', '__globals__', 'Botswana']
print(hsi["Botswana"].shape)  # 1476 x 256 x 145
print(hsi["Botswana"].min(), hsi["Botswana"].max())  # 0 45106
numpy.save(gsettings.datasets_folder_path / "Botswana" / "Botswana.npy", hsi["Botswana"])
