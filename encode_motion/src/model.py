import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import plot_3d_motion_frames_multiple, plot_3d_motion_animation, plot_3d_motion_frames_multiple
from glob import glob


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
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Invalid optimizer")
    return optimizer


# no activation class
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class TransformerMotionAutoencoder(pl.LightningModule):
    def __init__(
        self,
        config,

    ):
        super(TransformerMotionAutoencoder, self).__init__()
        self.input_length = config.get("input_length", 160)
        self.input_dim = config.get("input_dim", 96)
        self.hidden_dim = config.get("hidden_dim", 1024)
        self.n_layers = config.get("n_layers", 8)
        self.n_heads = config.get("n_heads", 6)
        self.dropout = config.get("dropout", 0.10)
        self.latent_dim = config.get("latent_dim", 256)
        self.lr = config.get("learning_rate", 1 * 1e-5)
        self.optimizer = config.get("optimizer", "AdamW")
        self.save_animations = config.get("save_animations", True)
        self.loss_function = CustomLoss(config.get("LOSS", {'mse': 1., 'klDiv': 0.000002, 'l1': 0.5}))
        self.load = config.get("load", False)
        self.checkpoint_path = config.get("checkpoint_path", None)
        self.activation = config.get("activation", "relu")
        self.activation = activation_dict[self.activation]
        self.transformer_activation = config.get("transformer_activation", "gelu")
        self.output_layer = config.get("output_layer", "linear")


        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.input_dim,
                nhead=self.n_heads,
                dim_feedforward=self.hidden_dim,
                dropout=self.dropout,
                batch_first=True,
                activation=self.transformer_activation,
                norm_first=False,
            ),
            num_layers=self.n_layers,
        )

        self.encoder_linear_block = nn.Sequential(
            nn.Linear(self.input_dim * self.input_length, self.hidden_dim*8),
            self.activation,
            nn.Linear(self.hidden_dim*8, self.hidden_dim*4),
            self.activation,
            nn.Linear(self.hidden_dim*4, 2 * self.hidden_dim),
            self.activation,
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            self.activation,
            nn.Linear(self.hidden_dim, 2 * self.latent_dim),
        )

        self.decoder_linear_block = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            self.activation,
            nn.Linear(self.hidden_dim, self.hidden_dim*2),
            self.activation,
            nn.Linear(self.hidden_dim*2, self.hidden_dim*4),
            self.activation,
            nn.Linear(self.hidden_dim*4, self.hidden_dim*8),
            self.activation,
            nn.Linear(self.hidden_dim*8, self.input_dim * self.input_length),
            self.activation,
        )

        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.input_dim,
                nhead=self.n_heads,
                dim_feedforward=self.hidden_dim,
                dropout=self.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=False,
            ),
            num_layers=self.n_layers,
        )


        # define out block
        if self.output_layer == "linear":
            # self.fc_out1 = nn.Linear(
            #     self.input_dim * self.input_length, self.input_dim * self.input_length
            # )
            self.output_block = nn.Sequential(
                # reshape: x = x.view(-1, self.input_dim * self.input_length)
                nn.Flatten(),
                nn.Linear(self.input_dim * self.input_length, self.input_dim * self.input_length),

            )
                
        elif self.output_layer == "transformer":
            ## now a decoder layer with no activation function and no normalization
            self.transformer_decoder_out = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=self.input_dim,
                    nhead=self.n_heads,
                    dim_feedforward=self.hidden_dim,
                    dropout=self.dropout,
                    batch_first=True,
                    activation=Identity(),
                    norm_first=False,
                ),
                num_layers=1,
            )
            self.output_block = nn.Sequential(
                self.transformer_decoder_out,
            )

        else:
            # identity
            self.output_block = nn.Sequential( Identity() )

        # 

        if self.load:
            print(f"Loading model from {self.checkpoint_path}")
            weights = torch.load(self.checkpoint_path)
            self.load_state_dict(weights['state_dict'])
            print('loaded model from:', self.checkpoint_path)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # print(x.shape)
        x = x.view(-1, self.input_length, self.input_dim)
        x = self.transformer_encoder(x)
        x = x.view(-1, self.input_dim * self.input_length)
        x = self.encoder_linear_block(x)
        mu, logvar = x.chunk(2, dim=1)

        z = self.reparametrize(mu, logvar)

        x = self.decoder_linear_block(z)

        x = x.view(-1, self.input_length, self.input_dim)
        x = self.transformer_decoder(x, x)
        x = self.output_block(x)
        x = x.view(-1, self.input_length, self.input_dim // 3, 3)
        return x, mu, logvar, z

    def training_step(self, batch, batch_idx):
        res = self._common_step(batch, batch_idx)

        
        loss = {k + '_trn': v for k, v in res['loss'].items()}
        self.log_dict(loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        
        # clip gradients --> do i do this here? # TODO
        # if self.clip:
        #     torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        return res['loss']['total']

    def validation_step(self, batch, batch_idx):
        res = self._common_step(batch, batch_idx)
        loss = {k + '_val': v for k, v in res['loss'].items()}
        self.log_dict(loss)
        if batch_idx == 0:
            recon = res['recon']
            x = res['x']
            im_arr = plot_3d_motion_frames_multiple([recon.cpu().detach().numpy(), x.cpu().detach().numpy()], ["recon", "true"], 
                                                    nframes=5, radius=2, figsize=(20,8), return_array=True)
            # print(im_arr.shape)
            self.logger.experiment.add_image("recon_vs_true", im_arr, global_step=self.global_step)


    def test_step(self, batch, batch_idx):
        res = self._common_step(batch, batch_idx)
        loss = {k + '_tst': v for k, v in res['loss'].items()}
        #self.log("test_loss", loss)
        # we want to add test loss final to the tensorboard
        self.log_dict(loss)

        if batch_idx == 0 and self.save_animations:
            recon = res['recon']
            print("Saving animations")
            folder = self.logger.log_dir
            plot_3d_motion_animation(recon[0].cpu().detach().numpy(), "recon", figsize=(10, 10), fps=20, radius=2, save_path=f"{folder}/recon.mp4")
        return loss

    def _common_step(self, batch, batch_idx):
        x = batch
        recon, mu, logvar, z = self(x)
        loss = self.loss_function(x, recon, mu=mu, logvar=logvar)
        return dict(
            loss=loss,
            x = x,
            recon = recon,
            z=z,
            mu=mu,
            logvar=logvar,
        )

    def configure_optimizers(self):
        # this is also where we would put the scheduler
        return get_optimizer(self, self.optimizer, self.lr)


if __name__ == "__main__":
    model = TransformerMotionAutoencoder(
        input_length=160,
        input_dim=96,
        hidden_dim=1024,
        n_layers=8,
        n_heads=6,
        dropout=0.10,
        latent_dim=256,
        loss_function="MSELoss + KL",
        learning_rate=1 * 1e-5,
        optimizer="AdamW",
        kl_weight=0.000001,
        save_animations=True,
        load=False,
        checkpoint_path=None,
    )
    x = torch.randn(128, 160, 96)
    recon, mu, logvar = model(x)
    print(recon.shape)