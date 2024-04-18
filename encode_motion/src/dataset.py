import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tiktoken

# DATASETS
class MotionDataset(Dataset):
    def __init__(
        self,
        file_list_path,
        path,
        sequence_length=42,
    ):
        super().__init__()  # just added this: maybe it will work
        self.path_text_enc = "../../data/HumanML3D/HumanML3D/texts_enc/simple/"
        self.idx2word = np.load(f"{self.path_text_enc}idx2word.npz", allow_pickle=True)[
            "arr_0"
        ]
        self.word2idx = np.load(f"{self.path_text_enc}word2idx.npz", allow_pickle=True)[
            "arr_0"
        ]

        self.sequence_length = sequence_length
        filenames = np.loadtxt(file_list_path, delimiter=",", dtype=str)
        self.filenames = [f"{path}/{f}.npy" for f in filenames]

        self.filenames_text_enc = [
            f"{self.path_text_enc}encodings/{f}.npy" for f in filenames
        ]

        # set up tiktoken tokenizer
        self.enc = tiktoken.encoding_for_model("gpt-4")

        print(len(self.filenames))

        self.max_text_len = 100

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        motion_seq = np.load(file_name)
        
        pose0 = motion_seq[:1]

        root_travel = motion_seq[:, :1, :]
        root_travel = root_travel - root_travel[:1]  # relative to the first frame
        motion_less_root = motion_seq - root_travel  # relative motion
        velocity = np.diff(motion_seq, axis=0)
        velocity_relative = np.diff(motion_less_root, axis=0)
    
        text_enc = np.load(self.filenames_text_enc[idx])

        return pose0, velocity_relative, root_travel, motion_seq, text_enc


class MotionDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.file_list_paths = cfg.get("file_list_paths")
        self.path = cfg.get("_motion_path")
        self.sequence_length = cfg.get("seq_len", 200)
        self.batch_size = cfg.get("batch_size", 128)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None) -> None:
        self.train_ds, self.val_ds, self.test_ds = [
            MotionDataset(self.file_list_paths[i], self.path, self.sequence_length)
            for i in ["_train", "_val", "_test"]
        ]

        self.idx2word = self.train_ds.idx2word

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
        )
