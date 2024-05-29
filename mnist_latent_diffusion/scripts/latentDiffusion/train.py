
from modules.dataModules import LatentSpaceDataModule
from modules.latentDiffusion import LatentDiffusionModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import shutil
import os
import matplotlib.pyplot as plt
import torch
import yaml
from modules.loss import VAE_Loss
import torch.nn as nn
import torch.optim as optim
from utils import latent_picker, load_latent
from utils import load_config

class Classifier(nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()
            self.fc1 = nn.Linear(16, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 128)
            self.fc4 = nn.Linear(128, 10)
            self.sigmoid = nn.Sigmoid()
            self.act = nn.LeakyReLU()

        def forward(self, x):
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x))
            x = self.act(self.fc3(x))
            x = self.fc4(x)
            return self.sigmoid(x)
        
class ClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, z, y):
        self.z = z
        self.y = y

        # embed y
        self.y = y
        
        self.y = self.y.to('mps')
        self.z = self.z.to('mps')

    def __len__(self):
        return len(self.z)
    
    def __getitem__(self, idx):
        return self.z[idx], self.y[idx]
    

def instatiate_classifier(scaler, z, dm):
    
    criterion_classifier = nn.BCELoss()
    classifier = Classifier().to('mps')
    try:
        classifier = torch.load(f'classifier.pth')
        print('loaded classifier')
    except:
        print('failed to load classifier, training')

        optimizer = optim.AdamW(classifier.parameters(), lr=1e-3)

        if scaler is None:
            z_scaled = z
        else:
            z_scaled = torch.tensor(scaler.transform(z.detach().cpu())).float()
        dataset = ClassifierDataset(dm.X, dm.y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

        for epoch in range(2):
            running_loss = 0
            for z_batch, y_batch in dataloader:
                # z_batch = z_batch.to(device)
                # y_batch = y_batch.to(device)
                optimizer.zero_grad()
                y_pred = classifier(z_batch)
                loss = criterion_classifier(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                print(f'Epoch {epoch} Loss: {loss.item()}', end='\r')
            print(f'Epoch {epoch} Loss: {running_loss/len(dataloader)}')

        torch.save(classifier, f'classifier.pth')
        print('done training classifier')
    return classifier

def train(build_mode=False):
    logger = TensorBoardLogger("logs/latentDiffusion/", name="train")

    # make logs directory
    if not os.path.exists(logger.log_dir): os.makedirs(logger.log_dir)

    # cp config file to the log directory
    shutil.copyfile('configs/config_LatentDiffusion.yaml', f"{logger.log_dir}/hparams.yaml")    
    config = load_config('LatentDiffusion')

    # load latent space
    data_version, version = latent_picker()
    # save version number
    with open(f'{logger.log_dir}/version.txt', 'w') as f:
        f.write(str(version))
    z, y,  autoencoder, projector, projection = load_latent(data_version)
    projection = projector.transform(z.detach().cpu().numpy())

    # set up data module
    dm = LatentSpaceDataModule(z, y, batch_size=config['TRAIN' if not build_mode else 'BUILD']['DATA']['BATCH_SIZE'], scale=config['TRAIN' if not build_mode else 'BUILD']['DATA']['SCALE'],)
    dm.setup()
    scaler = dm.scaler
    torch.save(scaler, f'{logger.log_dir}/scaler.pth')

    # loss
    criteria = VAE_Loss(config['TRAIN' if not build_mode else 'BUILD']['LOSS'])

    # classifier
    # classifier = instatiate_classifier(scaler, z, dm)
    classifier = None
    latent_dim = z.shape[1]
    config['TRAIN' if not build_mode else 'BUILD']['MODEL']['latent_dim'] = latent_dim
    # set up model
    print('Initializing model')
    model = LatentDiffusionModule(autoencoder=autoencoder, 
                                 scaler=scaler,
                                criteria=criteria,
                                classifier=None,
                                projector=projector,
                                projection=projection,
                                labels=y,
                                verbose=True if build_mode else False,
                                 **config['TRAIN' if not build_mode else 'BUILD']['MODEL'])
    
    # load
    

    if config['TRAIN']['MODEL']['LOAD']:
        name, num = logger.log_dir.split('/')[-1].split('_')
        log_dir = '/'.join(logger.log_dir.split('/')[:-1])

        model = LatentDiffusionModule.load_from_checkpoint(
            # logger.log_dir + '/' 
            log_dir + '/version_' + str(int(num)-1) + '/checkpoints/' +
            config['TRAIN' if not build_mode else 'BUILD']['MODEL']['LOAD'])
        print("LOADed model")

    # train
    print('Training model')
    trainer = pl.Trainer(logger=logger, **config['TRAIN' if not build_mode else 'BUILD']['TRAINER'])
    trainer.fit(model, dm)
    epochs_trained = trainer.current_epoch

    # test
    res = trainer.test(model, dm.test_dataloader())

    # logging hyperparameters
    logger.log_hyperparams(model.hparams, metrics=res[0])
    print("done")

    # save model
    #del model.classifier
    torch.save(model, f'{"/".join(logger.log_dir.split("/")[:-1])}//model.pth')

    print("model saved")
