import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
# from torch_geometric.loader import DataLoader as DataLoader_geometric

def rotate_around_y(joints, angle):
    """
    Rotate joints around y axis.

    Args:
    - joints (np.array): 3D joints (joints_num, 3)
    - angle (float): angle to rotate (radian)

    Returns:
    - rotated_joints (np.array): rotated 3D joints (joints_num, 3)
    """
    if angle == 0:
        return joints
    
    if angle == np.pi:
        return np.flip(joints, axis=0)

    rotation_matrix = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )

    rotated_joints = np.dot(joints, rotation_matrix)

    return rotated_joints

def rotate_poses_to_align(poses):
    hib1 = poses[:, 1, [0, 2]]
    adj = hib1[:, 0]
    opp = hib1[:, 1]
    y_positive = opp > 0
    angles = np.arctan(adj/opp) + np.pi * y_positive # smart input by alice!
    angles

    pose_rotated = [rotate_around_y(p, a) for p, a in zip(poses, angles)]
    pose_rotated = np.array(pose_rotated)
    # angles2 = np.zeros(len(angles))
    # hib2_y = pose_rotated[:, 2, 2]
    # angles2[hib2_y < 0] = np.pi

    return pose_rotated

class PoseDataset(Dataset):
    def __init__(self, file_list_path, motion_path, 
                 verbose=False, 
                 subtract_root=True,
                 rotate_to_align=True,
                 **kwargs):
        filenames = np.loadtxt(file_list_path, delimiter=',', dtype=str)
        self.filenames = [f'{motion_path}/{f}.npy' for f in filenames]
        tiny = kwargs.get('tiny', False)
        if tiny: self.filenames = self.filenames[:tiny]
        all_poses = [np.load(f) for f in self.filenames]
        

        # subtract root joint
        if subtract_root:
            for i in range(len(all_poses)):
                all_poses[i] -= all_poses[i][:, :1, :]
        example_video = all_poses[1]
        
        all_poses = np.concatenate(all_poses, axis=0)

        # rotate to align with x-axis
        if rotate_to_align:
            all_poses = rotate_poses_to_align(all_poses)
            example_video = rotate_poses_to_align(example_video)
        
        self.all_poses = torch.tensor(all_poses).float()
        self.example_video = torch.tensor(example_video).float()
        if verbose: print('len of all_poses:', len(self.all_poses))
        
    def __len__(self):
        return len(self.all_poses)

    def __getitem__(self, idx):
        pose = self.all_poses[idx]
        return pose
    
    def get_all_poses(self):
        return self.all_poses
   
class PoseDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128, num_workers=4, verbose=False, 
                 subtract_root=True,
                 rotate_to_align=True, **kwargs):
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
        self.subtract_root = subtract_root
        self.rotate_to_align = rotate_to_align


    def setup(self, stage) -> None:
        self.train_ds, self.val_ds, self.test_ds = [
            PoseDataset(f, self.path, self.verbose, 
                        tiny=self.tiny
                        , 
                 subtract_root=self.subtract_root,
                 rotate_to_align=self.rotate_to_align
                        ) for f in self.file_list_paths
        ]
        self.test_video = self.test_ds.example_video


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
