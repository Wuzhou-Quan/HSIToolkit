from typing import List, OrderedDict
import pathlib


class MultiFilePathProvider:
    """
    A class for managing multiple file paths in a directory.

    Attributes:
        root_path (pathlib.Path): The root path of the directory.
        valid_ext (List[str]): A list of valid file extensions to consider. Defaults to [".tif", ".mat"].
        img_name_list (List[str]): A list of image names with valid extensions in the directory.

    Methods:
        __getitem__(index: int) -> pathlib.Path: Returns the path of the image at the given index.
        __len__() -> int: Returns the number of images in the directory.

    """

    def __init__(self, root_path: pathlib.Path, valid_ext: List[str] = [".tif", ".mat"]):
        """
        Initializes a MultiFilePathManager instance.

        Args:
            root_path (pathlib.Path): The root path of the directory.
            valid_ext (List[str], optional): A list of valid file extensions to consider. Defaults to [".tif", ".mat"].
        """
        self.root_path = root_path
        self.img_name_list = [img.name for img in self.root_path.iterdir() if img.suffix.lower() in valid_ext]

    def __getitem__(self, index: int) -> pathlib.Path:
        """
        Returns the path of the image at the given index.

        Args:
            index (int): The index of the image.

        Returns:
            pathlib.Path: The path of the image.
        """
        img_path = self.root_path / self.img_name_list[index]
        return img_path

    def __len__(self) -> int:
        """
        Returns the number of images in the directory.

        Returns:
            int: The number of images.
        """
        return len(self.img_name_list)


class MultiSubsetPathProvider:
    def __init__(self, root_path: pathlib.Path, valid_ext: List[str] = [".tif", ".mat"], valid_subsets: List[str] = None):
        """
        Initializes a MultiSubsetsPathManager instance.

        Args:
            root_path (pathlib.Path): The root path of the directory.
            valid_ext (List[str], optional): A list of valid file extensions to consider. Defaults to [".tif", ".mat"].
            valid_subsets (List[str], optional): A list of valid subset names. Defaults to None.
        """
        self.subsets: OrderedDict[str, MultiFilePathProvider] = OrderedDict()
        for subset_name in valid_subsets or root_path.iterdir():
            subset_path = root_path / subset_name if isinstance(subset_name, str) else subset_name
            if not subset_path.is_dir():
                continue
            self.subsets[subset_path.name] = MultiFilePathProvider(subset_path, valid_ext)

    def __getitem__(self, index: int) -> pathlib.Path:
        """
        Returns the path of the image at the given index.

        Args:
            index (int): The index of the image.

        Returns:
            pathlib.Path: The path of the image.
        """
        for subset in self.subsets.values():
            if index < len(subset):
                return subset[index]
            index -= len(subset)
        raise IndexError("Index out of range")

    def __len__(self) -> int:
        """
        Returns the total number of images in all subsets.

        Returns:
            int: The total number of images.
        """
        return sum(len(subset) for subset in self.subsets.values())
