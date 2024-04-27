import torch
from torch import nn
import pytorch_lightning as pl
import torchvision
from modules.metrics import obtain_metrics
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import plotUMAP


activation_dict = {
    # soft step
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'softsign': nn.Softsign(),

    # rectifiers
    'leaky_relu': nn.LeakyReLU(),
    'ReLU': nn.ReLU(),
    'elu': nn.ELU(),
    'swish': nn.SiLU(),

    # identity
    'None': nn.Identity(),
    None: nn.Identity(),
}

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvLayer, self).__init__()
        self.verbose = kwargs.get('verbose', False)        
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
        shape_before = x.shape
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        if self.dropout > 0:
            x = nn.Dropout(self.dropout)(x)
        shape_after = x.shape
        if self.verbose:
            print(f"{'ConvLayer'.ljust(20)}: {tuple(shape_before)} \t-> {tuple(shape_after)}")
        return x
    
    def __repr__(self):
        return f'ConvLayer(in_channels={self.conv.in_channels}, out_channels={self.conv.out_channels}, kernel_size={self.conv.kernel_size}, stride={self.conv.stride}, padding={self.conv.padding})'
    
class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(LinearLayer, self).__init__()
        self.verbose = kwargs.get('verbose', False)
        self.linear = nn.Linear(in_features, out_features)
        self.act = kwargs.get('act', nn.ReLU())
        self.dropout = kwargs.get('dropout', 0.)
        
    def forward(self, x):
        shape_before = x.shape
        x = self.linear(x)
        x = self.act(x)
        if self.dropout > 0:
            x = nn.Dropout(self.dropout)(x)
        shape_after = x.shape
        if self.verbose:
            print(f"{'LinearLayer'.ljust(20)}: {tuple(shape_before)} \t-> {tuple(shape_after)}")
        return x

    def __repr__(self):
        return f'LinearLayer(in_features={self.linear.in_features}, out_features={self.linear.out_features})'

class ConvTransposeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvTransposeLayer, self).__init__()
        self.verbose = kwargs.get('verbose', False)
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
        shape_before = x.shape
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        if self.dropout > 0:
            x = nn.Dropout(self.dropout)(x)
        shape_after = x.shape
        if self.verbose:
            print(f"{'ConvTransposeLayer'.ljust(20)}: {tuple(shape_before)} \t-> {tuple(shape_after)}")
        return x

    def __repr__(self):
        return f'ConvTransposeLayer(in_channels={self.conv.in_channels}, out_channels={self.conv.out_channels}, kernel_size={self.conv.kernel_size}, stride={self.conv.stride}, padding={self.conv.padding})'

class VAE_cnn_old(pl.LightningModule):
    def __init__(self, **kwargs):
        """
        Latent dim fixed at 16
        """
        super().__init__()
        self.verbose = kwargs.get('verbose', False)
        self.act = kwargs.get('act', nn.ReLU())
        self.out_act = kwargs.get('out_act', nn.Identity())

        # self.batch_norm = nn.BatchNorm2d(1)

        self.encoder_conv_block = nn.Sequential(
            ConvLayer(1, 32, dropout=0.1, batch_norm=True, act=self.act, pool=False, verbose=self.verbose),
            ConvLayer(32, 64, dropout=0.0, batch_norm=True, act=self.act, pool=True, verbose=self.verbose),
            ConvLayer(64, 32, dropout=0.1, batch_norm=True, act=self.act, pool=True, verbose=self.verbose),
            ConvLayer(32, 16, dropout=0.0, batch_norm=True, act=self.act, pool=True, verbose=self.verbose)
        )

        self.encoder_linear_block = nn.Sequential(
            # flatten
            nn.Flatten(),
            LinearLayer(256, 128, dropout=0.0, act=self.act, verbose=self.verbose),
            LinearLayer(128, 64, dropout=0.0, act=self.act, verbose=self.verbose),
            LinearLayer(64, 32, dropout=0., act=nn.Identity(), verbose=self.verbose)
        )

        # decoder
        self.decoder_linear_block = nn.Sequential(
            LinearLayer(16, 32, dropout=0.0, act=self.act, verbose=self.verbose),
            LinearLayer(32, 64, dropout=0.0, act=self.act, verbose=self.verbose),
            LinearLayer(64, 128, dropout=0.0, act=self.act, verbose=self.verbose),
            LinearLayer(128, 256, dropout=0.0, act=self.act, verbose=self.verbose),
        )

        self.decoder_conv_block = nn.Sequential(
            # unflatten
            nn.Unflatten(1, (16, 4, 4)),
            ConvTransposeLayer(16, 32, kernel_size=3, stride=1, dropout=0.0, batch_norm=True, act=self.act, out_padding=0, verbose=self.verbose),
            ConvTransposeLayer(32, 16, kernel_size=3, stride=2, dropout=0.0, batch_norm=True, act=self.act, verbose=self.verbose),
            ConvTransposeLayer(16, 1, kernel_size=3, dropout=0.0, batch_norm=True, act=self.act, verbose=self.verbose, out_padding=1),           
        )

        # initialize weights as xavier
        self.encoder_conv_block.apply(self._init_weights)
        self.encoder_linear_block.apply(self._init_weights)
        self.decoder_linear_block.apply(self._init_weights)
        self.decoder_conv_block.apply(self._init_weights)

    def _init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.1)

    def encode(self, x):
        # pad with 2 zeros on each side
        x = F.pad(x, (2, 2, 2, 2), mode='constant', value=0)

        x = self.encoder_conv_block(x)
        
        x = self.encoder_linear_block(x)
        mu, logvar = x.chunk(2, dim=-1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        x = self.decoder_linear_block(z)
        
        x = self.decoder_conv_block(x)

        return x
    
    def forward(self, x):
        # apply batch norm
        # x = self.batch_norm(x)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_tilde = self.decode(z)
        return x_tilde, z,  mu, logvar

class VAE_cnn(pl.LightningModule):
    def __init__(self, **kwargs):
        """
        Latent dim fixed at 16
        """
        super().__init__()
        self.verbose = kwargs.get('verbose', False)
        self.act = kwargs.get('act', nn.ReLU())
        self.out_act = kwargs.get('out_act', nn.Identity())
        self.latent_drop_out_rate = kwargs.get('latent_drop_out_rate', 0.25)
        self.dropout = kwargs.get('dropout', 0.0)

        # self.batch_norm = nn.BatchNorm2d(1)

        self.encoder_conv_block = nn.Sequential(
            ConvLayer(1, 32, dropout=self.dropout, batch_norm=True, act=self.act, pool=False, verbose=self.verbose),
            ConvLayer(32, 64, dropout=self.dropout, batch_norm=True, act=self.act, pool=True, verbose=self.verbose),
            ConvLayer(64, 32, dropout=self.dropout, batch_norm=True, act=self.act, pool=True, verbose=self.verbose),
            ConvLayer(32, 16, dropout=self.dropout, batch_norm=True, act=self.act, pool=True, verbose=self.verbose)
        )

        self.encoder_linear_block = nn.Sequential(
            # flatten
            nn.Flatten(),
            LinearLayer(256, 128, dropout=self.dropout, act=self.act, verbose=self.verbose),
            LinearLayer(128, 64, dropout=self.dropout, act=self.act, verbose=self.verbose),
            LinearLayer(64, 32, dropout=self.dropout, act=nn.Identity(), verbose=self.verbose)
        )

        # we could look at the latent covariance matrix as a function of the dropout rate
        self.latent_dropouts = [
                nn.Dropout(self.latent_drop_out_rate),
                nn.Dropout(self.latent_drop_out_rate/2),
                nn.Dropout(0),
        ]  

        # decoder
        self.decoder_linear_block = nn.Sequential(
            LinearLayer(16, 32, dropout=self.dropout, act=self.act, verbose=self.verbose),
            LinearLayer(32, 64, dropout=self.dropout, act=self.act, verbose=self.verbose),
            LinearLayer(64, 128, dropout=self.dropout, act=self.act, verbose=self.verbose),
            LinearLayer(128, 256, dropout=self.dropout, act=self.act, verbose=self.verbose),
        )

        self.decoder_conv_block = nn.Sequential(
            # unflatten
            nn.Unflatten(1, (16, 4, 4)),
            ConvTransposeLayer(16, 32, kernel_size=3, stride=1, dropout=self.dropout, batch_norm=True, act=self.act, out_padding=0, verbose=self.verbose),
            ConvTransposeLayer(32, 16, kernel_size=3, stride=2, dropout=self.dropout, batch_norm=True, act=self.act, verbose=self.verbose),
            ConvTransposeLayer(16, 2, kernel_size=3, dropout=self.dropout, batch_norm=True, act=self.act, verbose=self.verbose, out_padding=1),           
        )

        self.out_block_kernel3 = nn.Sequential(
            ConvLayer(2, 2, kernel_size=3, dropout=self.dropout, batch_norm=True, act=self.act, pool=False, verbose=self.verbose),
        )

        self.out_block_kernel5 = nn.Sequential(
            ConvLayer(2, 2,kernel_size=5, dropout=self.dropout, batch_norm=True, act=self.act, pool=False, verbose=self.verbose, padding=2),
        )

        self.out_block_kernel7 = nn.Sequential(
            ConvLayer(2, 2, kernel_size=7, dropout=self.dropout, batch_norm=True, act=self.act, pool=False, verbose=self.verbose, padding=3),
        )

        self.linear_final_out = nn.Sequential(
             LinearLayer(8, 4, dropout=self.dropout, act=self.act, verbose=self.verbose),
                LinearLayer(4, 2, dropout=self.dropout, act=self.act, verbose=self.verbose),
                LinearLayer(2, 1, dropout=self.dropout, act=self.out_act, verbose=self.verbose),
        )

        # initialize weights as xavier
        self.encoder_conv_block.apply(self._init_weights)
        self.encoder_linear_block.apply(self._init_weights)
        self.decoder_linear_block.apply(self._init_weights)
        self.decoder_conv_block.apply(self._init_weights)
        self.out_block_kernel3.apply(self._init_weights)
        self.out_block_kernel5.apply(self._init_weights)
        self.out_block_kernel7.apply(self._init_weights)
        self.linear_final_out.apply(self._init_weights)


    def _init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.1)

    def encode(self, x):
        # pad with 2 zeros on each side
        x = F.pad(x, (2, 2, 2, 2), mode='constant', value=0)

        x = self.encoder_conv_block(x)
        
        x = self.encoder_linear_block(x)
        mu, logvar = x.chunk(2, dim=-1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        x = self.decoder_linear_block(z)
        out1 = self.decoder_conv_block(x)
        out_block_kernel3 = self.out_block_kernel3(out1)
        out_block_kernel5 = self.out_block_kernel5(out1)
        out_block_kernel7 = self.out_block_kernel7(out1)
        x = torch.cat([out1, out_block_kernel3, out_block_kernel5, out_block_kernel7], dim=1)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.linear_final_out(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    
    def forward(self, x):
        # apply batch norm
        # x = self.batch_norm(x)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        # z = self.latent_dropout(z)
        # pick random dropout
        z = self.latent_dropouts[torch.randint(0, len(self.latent_dropouts), (1,))](z)

        
        x_tilde = self.decode(z)
        return x_tilde, z,  mu, logvar


class VAE2(pl.LightningModule):
    def __init__(self,criterion, **kwargs):
        
        super().__init__()
        
        self.act_func = activation_dict[kwargs.get("ACTIVATION", 'ReLU')]
        self.out_act_func = activation_dict[kwargs.get("OUT_ACTIVATION", 'None')]
        self.verbose = kwargs.get('verbose', False)
        self.latent_drop_out_rate = kwargs.get('LATENT_DROP_OUT_RATE', 0.25)
        self.dropout = kwargs.get('DROPOUT', 0.0)
        self.model = VAE_cnn(verbose=self.verbose, act=self.act_func, out_act=self.out_act_func, latent_drop_out_rate=self.latent_drop_out_rate, dropout=self.dropout)

        self.lr = kwargs.get("LEARNING_RATE", 1e-3)
        self.criterion = criterion
        self.mul_KL_per_epoch = kwargs.get("MUL_KL_PER_EPOCH", 1)

        self.validation_step_outputs = {'x_hat' : [], 'z' : [], 'y' : []}
        self.train_step_outputs = {'x_hat' : [], 'z' : [], 'y' : []}

        self.test_step_outputs = []
        self.test_latents = []
        self.test_labels = []

        self.enable_UMAP = kwargs.get('ENABLE_UMAP', False)

    
    def forward(self, x):
        return self.model(x)
    
    def _common_step(self, batch, stage='train'):
        x, y = batch
        x_tilde, z, mu, logvar = self(x)
        loss_data = {
            'RECONSTRUCTION_L2': {'rec': x_tilde, 'true': x},
            'DIVERGENCE_KL': {'mu': mu, 'logvar': logvar}
        }
        # print('loss_data _common_step:', loss_data)
        total_loss, losses_scaled, losses_unscaled = self.criterion(loss_data)

        losses_scaled = {f'{stage}_{k}': v for k, v in losses_scaled.items()}
        losses_unscaled = {f'{stage}_unscaled_{k}': v for k, v in losses_unscaled.items()}
            
        return dict(
            loss=total_loss,
            losses_scaled=losses_scaled,
            losses_unscaled=losses_unscaled,
            x=x,
            x_hat=x_tilde,
            y=y,
            z=z,
            mu=mu,
            logvar=logvar,
        )
    
    def training_step(self, batch, batch_idx):
        res = self._common_step(batch, 'train')
        self.log('train_loss', res['loss'])
        #self.log_dict(res['losses_scaled'], prog_bar=True)
        self.log_dict(res['losses_unscaled'], prog_bar=True)

        if batch_idx <10:
            self.train_step_outputs['x_hat'].append(res['x_hat'])
            self.train_step_outputs['z'].append(res['z'])
            self.train_step_outputs['y'].append(res['y'])

        return res['loss']
    
    def on_train_epoch_end(self):
        # show images at the end of the epoch
        recs = torch.cat(self.train_step_outputs['x_hat'], dim=0)[:32]
        grid = torchvision.utils.make_grid(recs, nrow=8, normalize=True)
        self.logger.experiment.add_image('reconstruction_train', grid, self.current_epoch)
        self.train_step_outputs['x_hat'] = []

        z = torch.cat(self.train_step_outputs['z'], dim=0).view(-1, 16)
        # self.logger.experiment.add_histogram('z', z, self.current_epoch)
        y = torch.cat(self.train_step_outputs['y'], dim=0)
        if self.enable_UMAP:
            fig = plotUMAP(z, y, latent_dim=16, 
                        KL_weight=self.criterion.loss_weights['DIVERGENCE_KL'],
                        save_path=None, show=False, max_points=10000)
            self.logger.experiment.add_figure('UMAP', fig, self.current_epoch)
            plt.close(fig)

        # make covariance matrix of latent space and plot
        cov = torch.cov(z.T)
        cov_fig = plt.figure()
        plt.imshow(cov.cpu().detach().numpy())
        plt.colorbar()
        plt.title('Covariance matrix of latent space')
        self.logger.experiment.add_figure('Covariance matrix', cov_fig, self.current_epoch)

        self.train_step_outputs['y'] = []
        self.train_step_outputs['z'] = []

        # increase KL
        
    def validation_step(self, batch, batch_idx):
        res = self._common_step(batch, 'val')
        self.log('val_loss', res['loss'])
        #self.log_dict(res['losses_scaled'], prog_bar=True)
        self.log_dict(res['losses_unscaled'], prog_bar=False)
        self.validation_step_outputs['x_hat'].append(res['x_hat'])
        self.validation_step_outputs['z'].append(res['z'])
        self.validation_step_outputs['y'].append(res['y'])

    def on_validation_epoch_end(self):
        # show images at the end of the epoch
        
        recs = torch.cat(self.validation_step_outputs['x_hat'], dim=0)[:32]
        grid = torchvision.utils.make_grid(recs, nrow=8, normalize=True)
        self.logger.experiment.add_image('reconstruction_val', grid, self.current_epoch)
        self.validation_step_outputs['x_hat'] = []

        z = torch.cat(self.validation_step_outputs['z'], dim=0).view(-1, 16)
        # self.logger.experiment.add_histogram('z', z, self.current_epoch)
        y = torch.cat(self.validation_step_outputs['y'], dim=0)
        if self.enable_UMAP:
            # print('z shape:', z.shape, 'y shape:', y.shape)
            fig = plotUMAP(z, y, latent_dim=16, 
                        KL_weight=self.criterion.loss_weights['DIVERGENCE_KL'],
                            save_path=None, show=False, max_points=10000)
            self.logger.experiment.add_figure('UMAP', fig, self.current_epoch)
            plt.close(fig)

        # make covariance matrix of latent space and plot
        cov = torch.cov(z.T)
        cov_fig = plt.figure()
        plt.imshow(cov.cpu().detach().numpy())
        plt.colorbar()
        plt.title('Covariance matrix of latent space')
        self.logger.experiment.add_figure('Covariance matrix', cov_fig, self.current_epoch)

        
        self.validation_step_outputs['y'] = []
        self.validation_step_outputs['z'] = []

        # increase KL
        self.criterion.loss_weights['DIVERGENCE_KL'] *= self.mul_KL_per_epoch

    def test_step(self, batch, batch_idx):
        res = self._common_step(batch, 'test')
        self.log('test_loss', res['loss'])
        #self.log_dict(res['losses_scaled'], prog_bar=True)
        self.log_dict(res['losses_unscaled'], prog_bar=False)
        self.test_step_outputs.append(res['losses_unscaled']['test_unscaled_RECONSTRUCTION_L2'])

        latents = res['z']
        labels = res['y']
        self.test_latents.append(latents)
        self.test_labels.append(labels)

    def on_test_epoch_end(self):
        # run metrics on test latents
        latents = torch.cat(self.test_latents, dim=0)
        y = torch.cat(self.test_labels, dim=0)
        # convert to numpy
        latents = latents.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        latents = latents.reshape(latents.shape[0], -1)
        
        # calculate metrics
        metrics_res = obtain_metrics(latents, y)

        # log metrics
        #self.log_dict(metrics_res, prog_bar=False)
        self.metric_res = metrics_res
        return torch.stack(self.test_step_outputs).mean(), 

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.9)
        # increase lr by 2x
        # scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1.)

        # Increase KL weight by 10x

        return [optimizer],  [scheduler1]

class VAE_old(pl.LightningModule):
    def __init__(self, criterion, **kwargs):
        
        super().__init__()
        
        self.latent_dim = kwargs.get("LATENT_DIM")
        self.conv_channels = kwargs.get("CONV_CHANNELS")
        self.fc_units = kwargs.get("FC_UNITS")
        self.act_func = activation_dict[kwargs.get("ACTIVATION", 'ReLU')]
        self.out_act_func = activation_dict[kwargs.get("OUT_ACTIVATION", None)]

        self.lr = kwargs.get("LEARNING_RATE", 1e-3)

        self.criterion = criterion

        self.use_label_for_decoder = kwargs.get("USE_LABEL_FOR_DECODER", False)
        self.mul_KL_per_epoch = kwargs.get("MUL_KL_PER_EPOCH", 1)


        self.setup_blocks()

        self.validation_step_outputs = {'x_hat' : [], 'z' : [], 'y' : []}
        self.test_step_outputs = []

        self.target_embedding = nn.Embedding(10, 10)

        self.test_latents = []
        self.test_labels = []

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
        dim_z = self.latent_dim
        if self.use_label_for_decoder:
            dim_z += 10
        self.decoder_linear_block = nn.Sequential(
            LinearLayer(dim_z, fu[4], dropout=0.0, act=self.act_func),
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
    
    def decode(self, z, y=None):
        # random mask over z
        latent_drop_out_rate = 0.25
        mask = torch.rand_like(z) > latent_drop_out_rate
        z = z * mask

        if y is not None:
            # print('y:', y.shape)
            y_emb = self.target_embedding(y)
            z = torch.cat([z, y_emb], dim=1)
        x = self.decoder_linear_block(z)
        x = self.decoder_conv_block(x)
        x = self.output_linear_block(x)
        return x
    
    def forward(self, x, y=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), z,  mu, logvar
    
    def _common_step(self, batch, stage='train'):
        x, y = batch
        if self.use_label_for_decoder:

            x_hat, z, mu, logvar = self(x, y)
        else:
            x_hat, z, mu, logvar = self(x, None)
        loss_data = {
            'RECONSTRUCTION_L2': {'rec': x_hat, 'true': x},
            'DIVERGENCE_KL': {'mu': mu, 'logvar': logvar}
        }
        # print('loss_data _common_step:', loss_data)
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
        #self.log_dict(res['losses_scaled'], prog_bar=True)
        self.log_dict(res['losses_unscaled'], prog_bar=False)

        return res['loss']
    
    def validation_step(self, batch, batch_idx):
        res = self._common_step(batch, 'val')
        self.log('val_loss', res['loss'])
        #self.log_dict(res['losses_scaled'], prog_bar=True)
        self.log_dict(res['losses_unscaled'], prog_bar=False)
        self.validation_step_outputs['x_hat'].append(res['x_hat'])
        self.validation_step_outputs['z'].append(res['z'])
        self.validation_step_outputs['y'].append(res['y'])

    def on_validation_epoch_end(self):
        # show images at the end of the epoch
        
        recs = torch.cat(self.validation_step_outputs['x_hat'], dim=0)[:32]
        grid = torchvision.utils.make_grid(recs, nrow=8, normalize=True)
        self.logger.experiment.add_image('reconstruction', grid, self.current_epoch)
        self.validation_step_outputs['x_hat'] = []

        z = torch.cat(self.validation_step_outputs['z'], dim=0)
        # self.logger.experiment.add_histogram('z', z, self.current_epoch)
        y = torch.cat(self.validation_step_outputs['y'], dim=0)
        fig = plotUMAP(z, y, latent_dim=self.latent_dim, 
                       KL_weight=self.criterion.loss_weights['DIVERGENCE_KL'],
                          save_path=None, show=False)
        self.logger.experiment.add_figure('UMAP', fig, self.current_epoch)
        plt.close(fig)

        
        self.validation_step_outputs['y'] = []
        self.validation_step_outputs['z'] = []

        # increase KL
        self.criterion.loss_weights['DIVERGENCE_KL'] *= self.mul_KL_per_epoch

    def test_step(self, batch, batch_idx):
        res = self._common_step(batch, 'test')
        self.log('test_loss', res['loss'])
        #self.log_dict(res['losses_scaled'], prog_bar=True)
        self.log_dict(res['losses_unscaled'], prog_bar=False)
        self.test_step_outputs.append(res['losses_unscaled']['test_unscaled_RECONSTRUCTION_L2'])

        latents = res['z']
        labels = res['y']
        self.test_latents.append(latents)
        self.test_labels.append(labels)

    def on_test_epoch_end(self):
        # run metrics on test latents
        latents = torch.cat(self.test_latents, dim=0)
        y = torch.cat(self.test_labels, dim=0)
        # convert to numpy
        latents = latents.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        # calculate metrics
        metrics_res = obtain_metrics(latents, y)

        # log metrics
        #self.log_dict(metrics_res, prog_bar=False)
        self.metric_res = metrics_res
        return torch.stack(self.test_step_outputs).mean(), 

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.9)
        # increase lr by 2x
        # scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1.)

        # Increase KL weight by 10x

        return [optimizer],  [scheduler1]