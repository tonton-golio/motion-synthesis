import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import pandas as pd

try:
    from motion_latent_diffusion.modules import preprocessing as P
except ImportError:
    from modules import preprocessing as P


# DATASETS
class MotionDataset(Dataset):
    """Loads HumanML3D joint clips (T, 22, 3) and applies the corrected
    preprocessing pipeline: root-translation removal, optional heading
    canonicalization, per-axis normalization, and ZERO padding to ``seq_len``
    with a true length carried per sample (so padded frames are masked out of the
    model and loss — deep-review B14/B18).

    Normalization stats (mean/std) should be computed once on the training split
    and passed to the val/test datasets (and to the model for de-normalization).
    """

    def __init__(
        self,
        file_list_path,
        path,
        sequence_length=160,
        tiny=-1,
        mean=None,
        std=None,
        normalize=True,
        canonicalize=True,
    ):
        super().__init__()
        self.tiny = tiny
        self.sequence_length = sequence_length
        self.normalize = normalize
        self.canonicalize = canonicalize
        self.path_text_enc = "motion_latent_diffusion/data/CLIP/"

        filenames = np.loadtxt(file_list_path, delimiter=",", dtype=str)
        filenames = np.atleast_1d(filenames)
        self.filenames = [f"{path}/{f}.npy" for f in filenames]
        self.filenames_text_enc = [f"{self.path_text_enc}encodings/{f}.npy" for f in filenames]

        if self.tiny > 0:
            idxs = np.random.choice(len(self.filenames), self.tiny, replace=False)
            self.filenames = [self.filenames[i] for i in idxs]
            self.filenames_text_enc = [self.filenames_text_enc[i] for i in idxs]

        # load raw motion clips (each (T_i, 22, 3))
        raw = [torch.from_numpy(np.load(f)).float() for f in self.filenames]
        max_seq_len = max(len(m) for m in raw)

        # root-removal + heading canonicalization happen per-clip BEFORE stats.
        prepped = [P.remove_root_translation(m) for m in raw]
        if self.canonicalize:
            prepped = [P.canonicalize_heading(m) for m in prepped]

        # compute or accept normalization stats (per-axis x/y/z)
        if self.normalize:
            if mean is None or std is None:
                mean, std = P.compute_norm_stats(prepped)
            self.mean, self.std = mean, std
        else:
            self.mean = torch.zeros(3)
            self.std = torch.ones(3)

        # normalize valid frames then zero-pad, carrying the true length
        motion_seqs, lengths = [], []
        for m in prepped:
            if self.normalize:
                m = P.normalize(m, self.mean, self.std)
            padded, L = P.pad_to_length(m, sequence_length)
            motion_seqs.append(padded)
            lengths.append(L)
        self.motion_seqs = torch.stack(motion_seqs).float()
        self.lengths = torch.tensor(lengths, dtype=torch.long)

        size_mb = self.motion_seqs.element_size() * self.motion_seqs.nelement() / 1024 / 1024
        print(f"Number of sequences: {len(motion_seqs)}, max raw length: {max_seq_len}, "
              f"size: {size_mb:.2f} MB, normalized: {self.normalize}, canon: {self.canonicalize}")

        # text encodings
        text_encs = [np.load(f) for f in self.filenames_text_enc]
        self.text_encs = torch.from_numpy(np.array(text_encs)).float()

        # text group / action labels
        self.filenames_short = [f.split('/')[-1].split('.')[0] for f in self.filenames]
        self.file_nums = [int(f.split('/')[-1].split('.')[0].replace("M", "")) for f in self.filenames]
        path_grouped = 'motion_latent_diffusion/text_backup/texts_grouped.csv'
        df = pd.read_csv(path_grouped)
        self.df = df
        self.action_group = [df[df['fname'] == f + '.txt']['action_group_num'].values[0] for f in self.filenames_short]
        self.action = [df[df['fname'] == f + '.txt']['action_mapped_2_num'].values[0] for f in self.filenames_short]
        self.action_group = torch.from_numpy(np.array(self.action_group)).long()
        self.action = torch.from_numpy(np.array(self.action)).long()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        return (
            self.motion_seqs[idx],
            self.lengths[idx],
            self.text_encs[idx],
            self.action_group[idx],
            self.action[idx],
            self.file_nums[idx],
        )


class MotionDataModule1(pl.LightningDataModule):
    def __init__(self, **cfg):
        super().__init__()
        self.file_list_paths = cfg.get("file_list_paths")
        self.path = cfg.get("_motion_path")
        self.sequence_length = cfg.get("seq_len", 160)
        self.batch_size = cfg.get("batch_size", 128)
        self.tiny = cfg.get("tiny", -1)
        self.normalize = cfg.get("normalize", True)
        self.canonicalize = cfg.get("canonicalize_heading", True)
        self.data_mean = None
        self.data_std = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None) -> None:
        # Build train first so its stats normalize val/test (no train/val/test leakage).
        self.train_ds = MotionDataset(
            self.file_list_paths["_train"], self.path, self.sequence_length, self.tiny,
            normalize=self.normalize, canonicalize=self.canonicalize,
        )
        self.data_mean, self.data_std = self.train_ds.mean, self.train_ds.std
        self.val_ds, self.test_ds = [
            MotionDataset(
                self.file_list_paths[i], self.path, self.sequence_length, self.tiny,
                mean=self.data_mean, std=self.data_std,
                normalize=self.normalize, canonicalize=self.canonicalize,
            )
            for i in ["_val", "_test"]
        ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=4, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=2, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=2, persistent_workers=True)
