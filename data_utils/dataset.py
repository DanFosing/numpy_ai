from backend import xp

class Dataset:
    def __len__(self):
        raise NotImplementedError("Dataset subclasses should implement __len__")

    def __getitem__(self, index):
        raise NotImplementedError("Dataset subclasses should implement __getitem__")


class ArrayDataset(Dataset):

    def __init__(self, *array: xp.ndarray):
        if len(array) == 0:
            raise ValueError("ArrayDataset requires at least one array")
        n = array[0].shape[0]
        if not all(arr.shape[0] == n for arr in array):
            raise ValueError("All arrays must have the same first dimension")
        self.arrays = array
        self.num_samples = n

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return tuple(arr[index] for arr in self.arrays)
