
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tiktoken
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np


import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import glob
import torch



def pad(data, length=420):
    # Use numpy to handle padding and truncating efficiently
    if data.shape[0] < length:
        padding = np.pad(data, ((0, length - data.shape[0]), (0, 0), (0, 0)), mode='edge')
    else:
        padding = data[:length]
    return padding


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
            self.filenames = self.filenames[:1000]
            self.filenames_text_enc = self.filenames_text_enc[:1000]

        # load motion seqs
        motion_seqs = [np.load(f) for f in self.filenames]
        max_seq_len = max([len(m) for m in motion_seqs])
        motion_seqs = np.array([pad(m, sequence_length) for m in motion_seqs])
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



# DATASETS
class HumanML3D(Dataset):
    def __init__(self, t, max_text_len=100):
        self.file_names = glob.glob(f"../../data/data_fully_preprocessed/{t}/*.npz")

        print("loading", t)
        # print(self.file_names)
        self.max_text_len = max_text_len

        print(f"{t} len:", len(self.file_names))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx, verbose=False):
        if verbose:
            print(self.file_names[idx])
        data = np.load(self.file_names[idx], allow_pickle=True)
        motion = torch.tensor(data["motion"]).float()
        if verbose:
            print("motion:", motion.shape)
        velocity = torch.tensor(data["velocity"]).float()
        if verbose:
            print("velocity:", velocity.shape)
        rand_text_idx = np.random.choice(list(data["texts_encoded"].item().keys()))

        text_enc = torch.tensor(data["texts_encoded"].item()[rand_text_idx]).long()
        # zero pad
        if text_enc.shape[0] < self.max_text_len:
            text_enc = torch.cat(
                [
                    text_enc,
                    torch.zeros(
                        self.max_text_len - text_enc.shape[0], dtype=torch.long
                    ),
                ]
            )
        else:
            text_enc = text_enc[: self.max_text_len]

        if verbose:
            print("text_enc:", text_enc.shape)
        text = data["texts"].item()[rand_text_idx]

        # print(motion.shape, velocity.shape, text_enc.shape, text)
        if verbose:
            print()
        return motion, velocity, text_enc, text


class MotionDataModule2(pl.LightningDataModule):
    def __init__(self, **cfg):
        super().__init__()
        self.max_text_len = cfg.get("max_text_len", 100)
        self.sequence_length = cfg.get("seq_len", 120)
        self.batch_size = cfg.get("batch_size", 128)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None) -> None:
        self.train_ds, self.val_ds, self.test_ds = [
            HumanML3D(t, self.max_text_len) for t in ["train", "val", "test"]
        ]

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




# POSE

# DATASETS
class PoseDataset(Dataset):
    def __init__(self, file_list_path, motion_path, data_format='array', verbose=False):
        filenames = np.loadtxt(file_list_path, delimiter=',', dtype=str)
        self.filenames = [f'{motion_path}/{f}.npy' for f in filenames]
        
        all_poses = [np.load(f) for f in self.filenames]

        # subtract root joint
        for i in range(len(all_poses)):
            all_poses[i] -= all_poses[i][:, :1, :]

        self.all_poses = np.concatenate(all_poses, axis=0)
        if verbose: print('len of all_poses:', len(self.all_poses))
        self.data_format = data_format
        
    def __len__(self):
        return len(self.all_poses)

    def __getitem__(self, idx):
        pose = self.all_poses[idx]
        return pose
    
    def get_all_poses(self):
        return self.all_poses

        
class PoseDataModule(pl.LightningDataModule):
    def __init__(self, file_list_paths, path, data_format='array', batch_size=128, num_workers=4):
        super().__init__()
        self.file_list_paths = file_list_paths
        self.path = path
        self.data_format = data_format
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        pass

    def setup(self, stage) -> None:
        self.train_ds, self.val_ds, self.test_ds = [
            PoseDataset(
                self.file_list_paths[i], self.path, self.data_format
            )
            for i in ["train", "val", "test"]
        ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

