import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np


def pad_data(data, length=420):
    # print(data.shape)
    try:
        # Check if padding or truncation is needed based on the time steps dimension
        if data.shape[0] < length:
            # # Only pad the first dimension (time steps), keeping the rest unchanged
            # padding = ((0, length - data.shape[0]), (0, 0), (0, 0))
            # data = np.pad(data, padding, 'constant', constant_values=data[-1])
            padding = np.zeros(
                (length - data.shape[0], data.shape[1], data.shape[2]), dtype=np.float32
            )
            padding += data[-1]
            data = np.concatenate([data, padding], axis=0)
        elif data.shape[0] > length:
            data = data[:length]
        return data
    except Exception as e:
        print("data:", data)
        print("length:", length)
        print("data.shape:", data.shape)
        print(e)


# DATASETS
class MotionDataset(Dataset):
    def __init__(
        self,
        file_list_path,
        path,
        sequence_length=42,
    ):
        self.sequence_length = sequence_length
        filenames = np.loadtxt(file_list_path, delimiter=",", dtype=str)
        self.filenames = [f"{path}/{f}.npy" for f in filenames]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        motion_data = np.load(file_name)
        motion_data = pad_data(motion_data, self.sequence_length)
        return motion_data


class MotionDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.file_list_paths = cfg.get("file_list_paths")
        self.path = cfg.get("motion_path")
        self.sequence_length = cfg.get("seq_len", 200)
        self.batch_size = cfg.get("batch_size", 128)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage) -> None:
        self.train_ds, self.val_ds, self.test_ds = [
            MotionDataset(
                self.file_list_paths[i], self.path, self.sequence_length
            )
            for i in ["train", "val", "test"]
        ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=2, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=2, persistent_workers=True)

