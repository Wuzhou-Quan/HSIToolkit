# %%
import pathlib, numpy, gsettings
from inad_datasets.file_loader import file_loader

# %% Handle Pavia
# |- $gsettings.datasets_folder_path$
# |-- salinas
# |--- Salinas_corrected.mat
# |--+ [CREATED]Salinas_corrected.npy
spectral_file = pathlib.Path(gsettings.datasets_folder_path / "salinas" / "Salinas_corrected.mat")
hsi = file_loader[spectral_file.suffix.lower()](spectral_file)
print(hsi.keys())  # ['__header__', '__version__', '__globals__', 'salinas_corrected']
print(hsi["salinas_corrected"].shape)  # 512 x 217 x 204
numpy.save(gsettings.datasets_folder_path / "salinas" / "Salinas_corrected.npy", hsi["salinas_corrected"])
#%%
spectral_file = pathlib.Path(gsettings.datasets_folder_path / "salinas" / "Salinas_corrected.npy")
x = numpy.load(spectral_file)
print(x.min(),x.max())