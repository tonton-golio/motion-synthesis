import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from torch_geometric.data import Data

## instead of this, lets not pad with zeros, but instead pad with the last frame
def pad_data(data, length=420):
    # print(data.shape)
    try:
        # Check if padding or truncation is needed based on the time steps dimension
        if data.shape[0] < length:
            # # Only pad the first dimension (time steps), keeping the rest unchanged
            # padding = ((0, length - data.shape[0]), (0, 0), (0, 0))
            # data = np.pad(data, padding, 'constant', constant_values=data[-1])
            padding = np.zeros((length - data.shape[0], data.shape[1], data.shape[2]), dtype=np.float32)
            padding += data[-1]
            data = np.concatenate([data, padding], axis=0)
        elif data.shape[0] > length:
            data = data[:length]
        return data
    except Exception as e:
        print('data:', data)
        print('length:', length)
        print('data.shape:', data.shape)
        print(e)

# DATASETS
class PoseDataset(Dataset):
    def __init__(self, file_list_path, motion_path, data_format='array', verbose=False):
        filenames = np.loadtxt(file_list_path, delimiter=',', dtype=str)
        self.filenames = [f'{motion_path}/{f}.npy' for f in filenames]
        
        all_poses = [np.load(f) for f in self.filenames]
        self.all_poses = np.concatenate(all_poses, axis=0)
        if verbose: print('len of all_poses:', len(self.all_poses))
        self.data_format = data_format
        
    def __len__(self):
        return len(self.all_poses)

    # def make_graph(self, pose):
    #     # make graph
        
    #     pose = torch.tensor(pose, dtype=torch.float)
        
    #     data = Data(x=pose, edge_index=edge_index)
    #     return data

    def __getitem__(self, idx):
        pose = self.all_poses[idx]
        return pose
        if self.data_format == 'graph':
            pose = self.make_graph(pose)
            pose.validate(raise_on_error=True)
        
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

