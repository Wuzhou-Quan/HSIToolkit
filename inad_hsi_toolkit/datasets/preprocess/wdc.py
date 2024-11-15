import einops, numpy, pathlib
from ..file_loader import load_tif

spectral_file = pathlib.Path("/home/walterd/projects/mambahsi/data/WashintonDC/dc.tif")
data = (load_tif(spectral_file) + 2**15) / (2**16 - 1)
data = einops.rearrange(data,"c h w -> h w c")
print(data.shape)