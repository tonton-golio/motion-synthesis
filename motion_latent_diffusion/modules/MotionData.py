import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tiktoken
import torch
import glob
from utils import pad_crop

# DATASETS
class MotionDataset(Dataset):
    def __init__(
        self,
        file_list_path,
        path,
        sequence_length=42,
        tiny=False,
    ):
        super().__init__()  # just added this: maybe it will work
        self.tiny = tiny
        self.sequence_length = sequence_length
        # self.enc = tiktoken.encoding_for_model("gpt-4")
        # self.max_text_len = 100

        self.path_text_enc = "../stranger_repos/HumanML3D/HumanML3D/texts_enc/simple/"
        self.idx2word = np.load(f"{self.path_text_enc}idx2word.npz", allow_pickle=True)[ "arr_0"]
        self.word2idx = np.load(f"{self.path_text_enc}word2idx.npz", allow_pickle=True)["arr_0"]

        
        filenames = np.loadtxt(file_list_path, delimiter=",", dtype=str)
        self.filenames = [f"{path}/{f}.npy" for f in filenames]
        self.filenames_text_enc = [f"{self.path_text_enc}encodings/{f}.npy" for f in filenames]

        if self.tiny:
            n_tiny = 32
            # pick random
            idxs = np.random.choice(len(self.filenames), n_tiny, replace=False)
            self.filenames = [self.filenames[i] for i in idxs]
            self.filenames_text_enc = [self.filenames_text_enc[i] for i in idxs]
            

        # load motion seqs
        motion_seqs = [np.load(f) for f in self.filenames]
        max_seq_len = max([len(m) for m in motion_seqs])
        motion_seqs = np.array([pad_crop(m, sequence_length) for m in motion_seqs])
        self.motion_seqs = torch.from_numpy(motion_seqs).float()
        # check how large it is, in MB
        size_mb = self.motion_seqs.element_size() * self.motion_seqs.nelement() / 1024 / 1024
        print(f'Number of sequences: {len(motion_seqs)}, max length: {max_seq_len}, size: {size_mb:.2f} MB')

        # load text encodings
        text_encs = [np.load(f) for f in self.filenames_text_enc]
        self.text_encs = torch.from_numpy(np.array(text_encs)).long()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # file_name = self.filenames[idx]
        # motion_seq = np.load(file_name)
        # motion_seq = pad_data(motion_seq, self.sequence_length)
        motion_seq = self.motion_seqs[idx]

        text_enc = self.text_encs[idx]

        return motion_seq, text_enc

    def reconstruct(pose0, velocity_relative):
        motion_less_root = np.cumsum(
            np.concatenate([pose0, velocity_relative], dim=0), dim=0
        )
        return motion_less_root

class MotionDataModule1(pl.LightningDataModule):
    def __init__(self, **cfg):
        super().__init__()
        self.file_list_paths = cfg.get("file_list_paths")
        self.path = cfg.get("_motion_path")
        self.sequence_length = cfg.get("seq_len", 200)
        self.batch_size = cfg.get("batch_size", 128)
        self.tiny = cfg.get("tiny", False)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None) -> None:
        self.train_ds, self.val_ds, self.test_ds = [
            MotionDataset(self.file_list_paths[i], self.path, self.sequence_length, self.tiny)
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
