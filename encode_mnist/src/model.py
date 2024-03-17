# model for mnist autoencoder
# pytorch lightning

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch

class CustomLoss(nn.Module):
    def __init__(self, loss_dict):
        super(CustomLoss, self).__init__()
        self.loss_weights = {'mse': 0, 'L1': 0, 'klDiv': 0}
        for key, value in loss_dict.items():
            if key in ['mse', 'MSELoss']:
                self.loss_weights['mse'] = value
            elif key in ['l1', 'L1Loss']:
                self.loss_weights['L1'] = value
            elif key in ['kl', 'KL', 'KLLoss', 'klDiv']:
                self.loss_weights['klDiv'] = value
            else:
                raise ValueError(f'Loss function {key} not found')
    
    def forward(self, x, y, mu, logvar):
        dic = {}
        

        if self.loss_weights['mse'] > 0:
            dic['mse_us'] = F.mse_loss(x, y)
            dic['mse'] = self.loss_weights['mse'] * dic['mse_us']
        
        if self.loss_weights['L1'] > 0:
            dic['L1_us'] = F.l1_loss(x, y)
            dic['L1'] = self.loss_weights['L1'] * dic['L1_us']

        if self.loss_weights['klDiv'] > 0:
            dic['klDiv_us'] = self.kl_divergence(mu, logvar)
            dic['klDiv'] = dic['klDiv_us'] * self.loss_weights['klDiv']

        dic['total'] = sum(v for k, v in dic.items() if '_us' not in k)
        dic_us_total = {k:v for k, v in dic.items() if '_us' in k or k == 'total'}
        return dic_us_total
    
    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def get_optimizer(model, optimizer_name, learning_rate):
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError('Invalid optimizer')
    return optimizer

activation_dict = {
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'elu': nn.ELU(),
    'swish': nn.SiLU(),
    'mish': nn.Mish(),
    'softplus': nn.Softplus(),
    'softsign': nn.Softsign(),
    # 'bent_identity': nn.BentIdentity(),
    # 'gaussian': nn.Gaussian(),
    'softmax': nn.Softmax(),
    'softmin': nn.Softmin(),
    'softshrink': nn.Softshrink(),
    # 'sinc': nn.Sinc(),
}


# out put activation should be tunab

class Autoencoder(pl.LightningModule):
    def __init__(self, kw={}):
        
        super().__init__()
        self.latent_dim = kw.get("latent_dim", 2)
        nc = self.n_channels = kw.get("n_channels", [3, 32, 16])
        self.lin_size = kw.get("lin_size", 128)
        self.n_linear = kw.get("n_linear", 2)
        self.act_func = activation_dict[kw.get("activation", "relu")]
        self.dropout = kw.get("dropout", 0.1)
        self.batch_norm = kw.get("batch_norm", False)
        self.clip = kw.get("clip", False)
        # self.kernel_size = kw.get("kernel_size", 2)
        # self.stride = kw.get("stride", 2)
        self.lr = kw.get("learning_rate", 1e-3)
        self.optimizer = kw.get("optimizer", "Adam")
        self.bool = kw.get("bool", False)
        self.load = kw.get("load_model", False)
        self.path = kw.get("model_path", None)

        
        # print all those in self
        print('self:', '\n'.join(f'{k} : {v}' for k, v in self.__dict__.items() if not k.startswith("_")))

        
        self.loss_function = CustomLoss(kw.get("LOSS", {'mse': 1., 'klDiv': 0.000002, 'l1': 0.5}))

        # encoder
        self.encoder = nn.Sequential(
            # conv layers
            # nn.ConvTranspose2d(1, nc[0], kernel_size=3, stride=2, padding=1),
            # self.act_func,
            # nn.BatchNorm2d(nc[0]) if self.batch_norm else nn.Identity(),
            # nn.Dropout2d(self.dropout),

            nn.Conv2d(1, nc[0], kernel_size=3, stride=2, padding=0),
            self.act_func,
            nn.BatchNorm2d(nc[0]) if self.batch_norm else nn.Identity(),
            nn.Dropout2d(self.dropout),

            nn.Conv2d(nc[0], nc[1], 3, stride=2, padding=0),
            self.act_func,
            nn.BatchNorm2d(nc[1]) if self.batch_norm else nn.Identity(),
            nn.Dropout2d(self.dropout),

            # flatten layer
            nn.Flatten(),

            # linear layers
            nn.Linear(6*6*nc[-1], self.lin_size),
            self.act_func,
            nn.Dropout(self.dropout),
        )


        linear_encoder_block = [
            nn.Linear(self.lin_size, self.lin_size),
            self.act_func,
            nn.Dropout(self.dropout),
        ] * (self.n_linear - 1) if self.n_linear > 1 else []

        self.encoder = nn.Sequential(
            self.encoder,
            *linear_encoder_block,
        )

        self.embedder = nn.Linear(self.lin_size, self.latent_dim*2)


        # decoder
        linear_decoder_block = [
            nn.Linear(self.lin_size, self.lin_size),
            self.act_func,
            nn.Dropout(self.dropout),
        ] * (self.n_linear - 1) if self.n_linear > 1 else []


        self.decoder = nn.Sequential(
            # linear layers
            nn.Linear(self.latent_dim, self.lin_size),
            self.act_func,
            nn.Dropout(self.dropout),

            *linear_decoder_block,

            nn.Linear(self.lin_size, 8*8*nc[-1]),
            self.act_func,
            nn.Dropout(self.dropout),

            # reshape
            nn.Unflatten(1, (nc[-1], 8, 8)),
            
            # conv layers
            nn.ConvTranspose2d(nc[-1], nc[0], 2, 2, padding=1, output_padding=1),
            self.act_func,
            nn.BatchNorm2d(nc[0]) if self.batch_norm else nn.Identity(),
            nn.Dropout2d(self.dropout),

            nn.ConvTranspose2d(nc[0], 1,2, 2, padding=1, output_padding=0),
            
        )

        if not self.load:
            # initialize all weights as xavier
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0.01)

        else:
            weights = torch.load(self.path)
            self.load_state_dict(weights['state_dict'])
            print('loaded model from:', self.path)
            
            if "step=" in self.path:
                self.step0 = int(self.path.split("step=")[1].split('.')[0])

            

        

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encode(self, x):
        x = self.encoder(x)
        mu_logvar = self.embedder(x)
        mu = mu_logvar[:, :self.latent_dim]
        logvar = mu_logvar[:, self.latent_dim:]
        return mu, logvar
    
    def decode(self, z):
        x = self.decoder(z)
        if self.bool:
            x = nn.Sigmoid()(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x = self.decode(z)
        return x, mu, logvar, z

    def common_step(self, batch, batch_idx):
        x, y = batch
        # print(x.shape)
        # print(y.shape)
        x_hat, mu, logvar, z = self.forward(x)
        loss = self.loss_function(x, x_hat, mu, logvar)
        return dict(
            loss=loss,
            x=x,
            recon=x_hat,
            z=z,
            mu=mu,
            logvar=logvar,
            y=y
        )

    def training_step(self, batch, batch_idx):
        # loss, x, x_hat, z = self.common_step(batch, batch_idx)
        res = self.common_step(batch, batch_idx)
        loss = {k + '_trn': v for k, v in res['loss'].items()}
        self.log_dict(loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        
        # clip gradients --> do i do this here? # TODO
        # if self.clip:
        #     torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        return res['loss']['total']
    
    def validation_step(self, batch, batch_idx):
        res = self.common_step(batch, batch_idx)
        loss = {k + '_val': v for k, v in res['loss'].items()}
        # print('val losses:', loss)
        # print('val losses:', loss)
        # print('val losses:', loss)
        # print('val losses:', loss)
        self.log_dict(loss)#, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx == 0:
            print('logging images', end='\r')
            # grid = torchvision.utils.make_gri∂d(x, nrow=8, normalize=True)
            # self.logger.experiment.add_image('input', grid, 0)
            grid = torchvision.utils.make_grid(res['recon'][:32], nrow=8, normalize=True)
            self.logger.experiment.add_image('reconstruction', grid, global_step=self.global_step)#+self.step0)

            # plot latent space to logger
            
        
    def test_step(self, batch, batch_idx):
        res = self.common_step(batch, batch_idx)
        loss = {k + '_tst': v for k, v in res['loss'].items()}
        # self.log_dict(loss)#, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # if batch_idx == 0:
        #     self.logger.experiment.add_embedding(res['z'], metadata=res['y'], global_step=self.global_step)
        if batch_idx == 0:
            print('logging images', end='\r')
            # grid = torchvision.utils.make_gri∂d(x, nrow=8, normalize=True)
            # self.logger.experiment.add_image('input', grid, 0)
            grid = torchvision.utils.make_grid(res['recon'][:32], nrow=8, normalize=True)
            self.logger.experiment.add_image('reconstruction', grid, global_step=self.global_step)

            # plot latent space to logger

        return loss['total_tst'].item()
        
    def configure_optimizers(self):
        # this is also where we would put the scheduler
        return get_optimizer(self, self.optimizer, self.lr)