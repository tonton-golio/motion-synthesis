import torch
from torch import nn
import pytorch_lightning as pl
import torchvision
import os

activation_dict = {
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'ReLU': nn.ReLU(),
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
    'None': nn.Identity(),
    # 'sinc': nn.Sinc(),
}

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvLayer, self).__init__()
        
        kernel_size = kwargs.get('kernel_size', 3)
        stride = kwargs.get('stride', 1)
        padding = kwargs.get('padding', 1)
        batch_norm = kwargs.get('batch_norm', True)
        pool = kwargs.get('pool', True)
    
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.act = kwargs.get('act', nn.ReLU())
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()
        self.dropout = kwargs.get('dropout', 0.)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        if self.dropout > 0:
            x = nn.Dropout(self.dropout)(x)
        return x
    
    def __repr__(self):
        return f'ConvLayer(in_channels={self.conv.in_channels}, out_channels={self.conv.out_channels}, kernel_size={self.conv.kernel_size}, stride={self.conv.stride}, padding={self.conv.padding})'
    
class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(LinearLayer, self).__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        self.act = kwargs.get('act', nn.ReLU())
        self.dropout = kwargs.get('dropout', 0.)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        if self.dropout > 0:
            x = nn.Dropout(self.dropout)(x)
        return x

    def __repr__(self):
        return f'LinearLayer(in_features={self.linear.in_features}, out_features={self.linear.out_features})'

class ConvTransposeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvTransposeLayer, self).__init__()
        
        kernel_size = kwargs.get('kernel_size', 2)
        stride = kwargs.get('stride', 2)
        padding = kwargs.get('padding', 0)
        out_padding = kwargs.get('out_padding', 0)
        batch_norm = kwargs.get('batch_norm', True)
    
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, out_padding)
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.act = kwargs.get('act', nn.ReLU())

        self.dropout = kwargs.get('dropout', 0.)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        if self.dropout > 0:
            x = nn.Dropout(self.dropout)(x)
        return x

    def __repr__(self):
        return f'ConvTransposeLayer(in_channels={self.conv.in_channels}, out_channels={self.conv.out_channels}, kernel_size={self.conv.kernel_size}, stride={self.conv.stride}, padding={self.conv.padding})'

class VAE(pl.LightningModule):
    def __init__(self, criterion, **kwargs):
        
        super().__init__()
        
        self.latent_dim = kwargs.get("LATENT_DIM")
        self.conv_channels = kwargs.get("CONV_CHANNELS")
        self.fc_units = kwargs.get("FC_UNITS")
        self.act_func = activation_dict[kwargs.get("ACTIVATION", 'ReLU')]
        self.out_act_func = activation_dict[kwargs.get("OUT_ACTIVATION", None)]

        self.lr = kwargs.get("LEARNING_RATE", 1e-3)

        self.criterion = criterion
        self.setup_blocks()

        self.validation_step_outputs = {'x_hat' : [], 'z' : []}
        self.test_step_outputs = []

    def setup_blocks(self):
        cc = self.conv_channels
        fu = self.fc_units
        # encoder
        self.encoder_conv_block = nn.Sequential(
            ConvLayer(1, cc[0], dropout=0.05, batch_norm=True, act=self.act_func),
            ConvLayer(cc[0], cc[1], dropout=0., batch_norm=True, act=self.act_func),
            ConvLayer(cc[1], cc[2], dropout=0., batch_norm=True, pool=False, act=self.act_func)
        )

        self.encoder_linear_block = nn.Sequential(
            nn.Flatten(),
            LinearLayer(cc[2]*7*7, fu[0], dropout=0.1, act=self.act_func),
            LinearLayer(fu[0], fu[1], dropout=0., act=self.act_func),
            LinearLayer(fu[1], fu[2], dropout=0.0, act=self.act_func),
            LinearLayer(fu[2], fu[3], dropout=0.0, act=self.act_func),
            LinearLayer(fu[3], fu[4], dropout=0.0, act=self.act_func),
            LinearLayer(fu[4], self.latent_dim*2, dropout=0., act=nn.Identity())
        )

        # decoder
        self.decoder_linear_block = nn.Sequential(
            LinearLayer(self.latent_dim, fu[4], dropout=0.0, act=self.act_func),
            LinearLayer(fu[4], fu[3], dropout=0.0, act=self.act_func),
            LinearLayer(fu[3], fu[2], dropout=0.0, act=self.act_func),
            LinearLayer(fu[2], fu[1], dropout=0.05, act=self.act_func),
            LinearLayer(fu[1], fu[0], dropout=0.05, act=self.act_func),
            LinearLayer(fu[0], cc[2]*7*7, dropout=0.1, act=self.act_func),
            nn.Unflatten(-1, (cc[2], 7, 7))
        )

        self.decoder_conv_block = nn.Sequential(
            ConvTransposeLayer(cc[2], cc[1], dropout=0., batch_norm=True, act=self.act_func),
            ConvTransposeLayer(cc[1], cc[0], dropout=0., batch_norm=True, act=self.act_func),
            ConvTransposeLayer(cc[0], 1, dropout=0.05, batch_norm=True, kernel_size=3, stride=1, padding=1, act=self.act_func)
        )

        self.output_linear_block = nn.Sequential(
            nn.Flatten(),
            LinearLayer(28*28, 28*28, dropout=0.0, act=self.out_act_func),
            nn.Unflatten(-1, (1, 28, 28))
        )

        # initialize weights as xavier
        self.encoder_conv_block.apply(self._init_weights)
        self.encoder_linear_block.apply(self._init_weights)
        self.decoder_linear_block.apply(self._init_weights)
        self.decoder_conv_block.apply(self._init_weights)
        self.output_linear_block.apply(self._init_weights)

    def _init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.1)

    def encode(self, x):
        x = self.encoder_conv_block(x)
        x = self.encoder_linear_block(x)
        mu, logvar = x.chunk(2, dim=1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        x = self.decoder_linear_block(z)
        x = self.decoder_conv_block(x)
        x = self.output_linear_block(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), z,  mu, logvar
    
    def _common_step(self, batch, stage='train'):
        x, y = batch
        x_hat, z, mu, logvar = self(x)

        loss_data = {
            'RECONSTRUCTION_L2': {'rec': x_hat, 'true': x},
            'DIVERGENCE_KL': {'mu': mu, 'logvar': logvar}
        }

        total_loss, losses_scaled, losses_unscaled = self.criterion(loss_data)

        losses_scaled = {f'{stage}_{k}': v for k, v in losses_scaled.items()}
        losses_unscaled = {f'{stage}_unscaled_{k}': v for k, v in losses_unscaled.items()}
            
        return dict(
            loss=total_loss,
            losses_scaled=losses_scaled,
            losses_unscaled=losses_unscaled,
            x=x,
            x_hat=x_hat,
            y=y,
            z=z,
            mu=mu,
            logvar=logvar,
        )
    
    def training_step(self, batch, batch_idx):
        res = self._common_step(batch, 'train')
        self.log('train_loss', res['loss'])
        self.log_dict(res['losses_scaled'], prog_bar=True)
        self.log_dict(res['losses_unscaled'], prog_bar=False)

        return res['loss']
    
    def validation_step(self, batch, batch_idx):
        res = self._common_step(batch, 'val')
        self.log('val_loss', res['loss'])
        self.log_dict(res['losses_scaled'], prog_bar=True)
        self.log_dict(res['losses_unscaled'], prog_bar=False)
        self.validation_step_outputs['x_hat'].append(res['x_hat'])
        self.validation_step_outputs['z'].append(res['z'])

    def on_validation_epoch_end(self):
        # show images at the end of the epoch
        
        recs = torch.cat(self.validation_step_outputs['x_hat'], dim=0)[:32]
        grid = torchvision.utils.make_grid(recs, nrow=8, normalize=True)
        self.logger.experiment.add_image('reconstruction', grid, self.current_epoch)
        self.validation_step_outputs['x_hat'] = []

        z = torch.cat(self.validation_step_outputs['z'], dim=0)
        self.logger.experiment.add_histogram('z', z, self.current_epoch)

        self.validation_step_outputs['z'] = []

    def test_step(self, batch, batch_idx):
        res = self._common_step(batch, 'test')
        self.log('test_loss', res['loss'])
        self.log_dict(res['losses_scaled'], prog_bar=True)
        self.log_dict(res['losses_unscaled'], prog_bar=False)
        self.test_step_outputs.append(res['losses_unscaled']['test_unscaled_RECONSTRUCTION_L2'])

    def on_test_epoch_end(self):
        return torch.stack(self.test_step_outputs).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]