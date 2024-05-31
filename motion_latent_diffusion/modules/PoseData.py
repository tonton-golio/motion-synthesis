import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
# from torch_geometric.loader import DataLoader as DataLoader_geometric

class PoseDataset(Dataset):
    def __init__(self, file_list_path, motion_path, verbose=False, **kwargs):
        filenames = np.loadtxt(file_list_path, delimiter=',', dtype=str)
        self.filenames = [f'{motion_path}/{f}.npy' for f in filenames]
        tiny = kwargs.get('tiny', False)
        if tiny: self.filenames = self.filenames[:tiny]
        all_poses = [np.load(f) for f in self.filenames]

        # subtract root joint
        for i in range(len(all_poses)):
            all_poses[i] -= all_poses[i][:, :1, :]

        self.all_poses = np.concatenate(all_poses, axis=0)
        if verbose: print('len of all_poses:', len(self.all_poses))
        
    def __len__(self):
        return len(self.all_poses)

    def __getitem__(self, idx):
        pose = self.all_poses[idx]
        return pose
    
    def get_all_poses(self):
        return self.all_poses
   
class PoseDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128, num_workers=4, verbose=False, **kwargs):
        super().__init__()
        self.file_list_paths = [
            "../stranger_repos/HumanML3D/HumanML3D/train_cleaned.txt",
            "../stranger_repos/HumanML3D/HumanML3D/val_cleaned.txt",
            "../stranger_repos/HumanML3D/HumanML3D/test_cleaned.txt",
        ]
        self.path = "../stranger_repos/HumanML3D/HumanML3D/new_joints"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.verbose = verbose
        self.tiny = kwargs.get('tiny', False)


    def setup(self, stage) -> None:
        self.train_ds, self.val_ds, self.test_ds = [
            PoseDataset(f, self.path, self.verbose, tiny=self.tiny) for f in self.file_list_paths
        ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
