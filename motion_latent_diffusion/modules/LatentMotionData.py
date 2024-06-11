import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from utils import print_header
# transform
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

class LatentMotionData(pl.LightningDataModule):
    def __init__(self, z_train, z_val, z_test, texts_train, texts_val, texts_test, file_num_train, file_num_val, file_num_test, batch_size=32):
        super().__init__()
        # print('z shape:', z.shape)  # this is (n, z_dim)
        # print('texts shape:', texts.shape) # this is (n, 3, text_len) -> 3 is for the 3 different texts


        # self.z = z.repeat(3, 1)  # repeat z for each text
        # self.texts = texts.view(-1, texts.shape[-1])  # flatten texts
        # # print('z shape:', self.z.shape)  # this is (n*3, z_dim)
        # # print('texts shape:', self.texts.shape) # this is (n*3, text_len)
        # self.batch_size = batch_size
        # self.latent_dim = z.shape[-1]

        self.z_train = z_train.repeat(3, 1).cpu().numpy()
        self.z_val = z_val.repeat(3, 1).cpu().numpy()
        self.z_test = z_test.repeat(3, 1).cpu().numpy()

        self.texts_train = texts_train.view(-1, texts_train.shape[-1])
        self.texts_val = texts_val.view(-1, texts_val.shape[-1])
        self.texts_test = texts_test.view(-1, texts_test.shape[-1])

        self.file_num_train = file_num_train.repeat(3).cpu().numpy()
        self.file_num_val = file_num_val.repeat(3).cpu().numpy()
        self.file_num_test = file_num_test.repeat(3).cpu().numpy()

        for z in [self.z_train, self.z_val, self.z_test]:
            print_header('z')
            print(f"""
                    z shape: {z.shape}
                    z mean: {z.mean()}
                    z std: {z.std()}
                    z min: {z.min()}
                    z max: {z.max()}
                    """)

        scaler = StandardScaler()
        # scaler = MaxAbsScaler()
        self.z_train = scaler.fit_transform(self.z_train)
        self.z_val = scaler.transform(self.z_val)
        self.z_test = scaler.transform(self.z_test)
        
        self.z_train = torch.tensor(self.z_train).float().to('mps')
        self.z_val = torch.tensor(self.z_val).float().to('mps')
        self.z_test = torch.tensor(self.z_test).float().to('mps')

        self.file_num_train = torch.tensor(self.file_num_train).long()
        self.file_num_val = torch.tensor(self.file_num_val).long()
        self.file_num_test = torch.tensor(self.file_num_test).long()


        for z in [self.z_train, self.z_val, self.z_test]:
            print_header('z')
            print(f"""
                    z shape: {z.shape}
                    z mean: {z.mean()}
                    z std: {z.std()}
                    z min: {z.min()}
                    z max: {z.max()}
                    """)


        self.scaler = scaler
        self.batch_size = batch_size

        print('z_train shape:', self.z_train.shape)  # this is (n, z_dim)
        print('texts_train shape:', self.texts_train.shape)
        print('file_num_train shape:', self.file_num_train.shape)

        self.latent_dim = z_train.shape[-1]
   
    def setup(self, stage=None):
        
        # split data into train, val, test
        self.dataset_train = TensorDataset(self.z_train, self.texts_train, self.file_num_train)
        self.dataset_val = TensorDataset(self.z_val, self.texts_val, self.file_num_val)
        self.dataset_test = TensorDataset(self.z_test, self.texts_test, self.file_num_test)


    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False)
    