import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np

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

