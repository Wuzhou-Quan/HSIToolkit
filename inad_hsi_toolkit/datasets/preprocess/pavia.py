# %%
import pathlib, numpy, gsettings
from inad_datasets.file_loader import file_loader

# %% Handle Pavia
# |- $gsettings.datasets_folder_path$
# |-- pavia
# |--- Pavia.mat
# |--+ [CREATED]Pavia.npy
spectral_file = pathlib.Path(gsettings.datasets_folder_path / "pavia" / "Pavia.mat")
hsi = file_loader[spectral_file.suffix.lower()](spectral_file)
print(hsi.keys())  # ['__header__', '__version__', '__globals__', 'pavia']
print(hsi["pavia"].shape)  # 1096 x 715 x 102
print(hsi["pavia"].min(),hsi["pavia"].max())  #
numpy.save(gsettings.datasets_folder_path / "pavia" / "Pavia.npy", hsi["pavia"])

# %% Handle PaviaU
# |- $gsettings.datasets_folder_path$
# |-- pavia
# |--- PaviaU.mat
# |--+ [CREATED]PaviaU.npy
spectral_file = pathlib.Path(gsettings.datasets_folder_path / "pavia" / "PaviaU.mat")
hsi = file_loader[spectral_file.suffix.lower()](spectral_file)
print(hsi.keys())  # ['__header__', '__version__', '__globals__', 'paviaU']
print(hsi["paviaU"].shape)  # 610 x 340 x 103
print(hsi["paviaU"].min(),hsi["paviaU"].max())  # 
numpy.save(gsettings.datasets_folder_path / "pavia" / "PaviaU.npy", hsi["paviaU"])