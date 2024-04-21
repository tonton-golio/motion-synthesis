import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tiktoken


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
        motion_seq = pad_data(motion_seq, self.sequence_length)
        # pose0 = motion_seq[:1]

        # root_travel = motion_seq[:, :1, :]
        # root_travel = root_travel - root_travel[:1]  # relative to the first frame
        # motion_less_root = motion_seq - root_travel  # relative motion
        # velocity = np.diff(motion_seq, axis=0)
        # velocity_relative = np.diff(motion_less_root, axis=0)
        # print('velocity:', velocity.shape)
        # print('pose0:', pose0.shape)
        # pose0_and_velocity = np.concatenate([pose0, velocity], axis=0) # (seq_len, joints_num, 3)
        # pose0_and_velocity_relative = np.concatenate([pose0, velocity_relative], axis=0) # (seq_len, joints_num, 3)
        # print('velocity:', velocity.shape)

        # now load texts
        # with open(self.filenames_text[idx], 'r') as f:
        #     texts = f.read().split('\n')
        #     if type(texts) == str:
        #         texts = [texts]

        #     texts = texts[:3]

        #     if len(texts) <3:
        #         texts = texts + texts[-1:]

        #     # encode
        #     text_encs = [self.enc.encode(text) for text in texts]

        #     # pad
        #     for i, text_enc in enumerate(text_encs):
        #         if len(text_enc) < self.max_text_len:
        #             text_encs[i] += [0] * (self.max_text_len - len(text_enc))
        #         else:
        #             text_encs[i] = text_enc[:self.max_text_len]

        #     text_enc = np.array(text_encs).reshape(1, -1, self.max_text_len)

        # print('text:', texts)
        # print('text_enc:', text_enc.shape)

        text_enc = np.load(self.filenames_text_enc[idx])

        return motion_seq, text_enc

    def reconstruct(pose0, velocity_relative):
        motion_less_root = np.cumsum(
            np.concatenate([pose0, velocity_relative], dim=0), dim=0
        )
        return motion_less_root


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
