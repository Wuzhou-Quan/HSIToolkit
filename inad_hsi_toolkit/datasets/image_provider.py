import pathlib
from typing import Union
from .path_provider import MultiSubsetPathProvider, MultiFilePathProvider
from .file_loader import file_loader


class ImageProvider:
    def __init__(self, path_provider: Union[MultiSubsetPathProvider, MultiFilePathProvider], augments=None):
        self.path_manager = path_provider
        self.length = len(self.path_manager)
        self.aug = augments

    def __getitem__(self, index: int):
        file_path: pathlib.Path = self.path_manager[index]
        return file_loader[file_path.suffix.lower()](file_path)

    def __len__(self):
        return self.length
    
