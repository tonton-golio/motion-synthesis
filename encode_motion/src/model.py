import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import plot_3d_motion_frames_multiple, plot_3d_motion_animation, plot_3d_motion_frames_multiple
from glob import glob
import matplotlib.pyplot as plt

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
    def __init__(self, loss_weights):
        super(CustomLoss, self).__init__()
        self.loss_weights = loss_weights

    
    def forward(self, loss_data, mu, logvar):
        """
        Should be called like this:
        loss_data = {
            'velocity' : {'true': vel, 'rec': vel_rec, 'weight': 1},      
            'root' : {'true': root, 'rec': root_rec, 'weight': 1},      
            'pose0' : {'true': pose0, 'rec': pose0_rec, 'weight': 1},    
        }
        loss = self.loss_function(loss_data, mu, logvar)

        return loss, with loss['total'] as the total loss
        """
        loss = {}
        total_loss = 0
        for key, data in loss_data.items():
            if self.loss_weights[key] == 0:
                continue
            # l2
            loss[key] = F.mse_loss(data['rec'], data['true']) * self.loss_weights[key]
            # l1
            # loss[key] = F.l1_loss(data['rec'], data['true']) * self.loss_weights[key]
            total_loss += loss[key] 


        kl_loss = self.kl_divergence(mu, logvar)* self.loss_weights['kl_div']
        total_loss += kl_loss 
        loss['kl_divergence'] = kl_loss
        loss['total'] = total_loss
        return loss
    
        
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


class Decoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, seq_len, input_dim, n_layers, n_heads, dropout, hidden_dim_trans, transformer_activation, activation):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.hidden_dim_trans = hidden_dim_trans
        self.transformer_activation = transformer_activation
        self.activation = activation
        

        # # print all the inputs
        # print('hidden_dim:', hidden_dim)
        # print('latent_dim:', latent_dim)
        # print('seq_len:', seq_len)
        # print('input_dim:', input_dim)
        # print('n_layers:', n_layers)
        # print('n_heads:', n_heads)
        # print('dropout:', dropout)
        # print('hidden_dim_trans:', hidden_dim_trans)
        # print('transformer_activation:', transformer_activation)
        # print('activation:', activation)


        self.decoder_linear_block = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim*4),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim*4, self.hidden_dim*8),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim*8, self.input_dim * self.seq_len),
            self.activation,
        )

        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.input_dim,
                nhead=self.n_heads,
                dim_feedforward=self.hidden_dim_trans,
                dropout=self.dropout,
                batch_first=True,
                activation=self.transformer_activation,
                norm_first=True,
            ),
            num_layers=self.n_layers,
        )

        self.output_block = nn.Sequential(
            # reshape: x = x.view(-1, self.input_dim * self.seq_len)
            nn.Flatten(),
            nn.Linear(self.input_dim * self.seq_len, self.input_dim * self.seq_len),
        )

    def forward(self, z):
        x = self.decoder_linear_block(z)
        x = x.view(-1, self.seq_len, self.input_dim)
        x = self.transformer_decoder(x, x)
        # if self.output_layer == "transformer":
        #     x = self.output_block(x, x)
        # else:
        x = self.output_block(x)
        x = x.view(-1, self.seq_len, self.input_dim // 3, 3)
        return x


class TransformerMotionAutoencoder(pl.LightningModule):
    def __init__(
        self,
        config,

    ):
        super(TransformerMotionAutoencoder, self).__init__()
        self.seq_len = config.get("seq_len", 160)
        self.input_dim = config.get("input_dim", 66)
        self.hidden_dim = config.get("hidden_dim", 1024)
        self.n_layers = config.get("n_layers", 8)
        self.n_heads = config.get("n_heads", 6)
        self.dropout = config.get("dropout", 0.10)
        self.latent_dim = config.get("latent_dim", 256)
        self.lr = config.get("learning_rate", 1 * 1e-5)
        self.optimizer = config.get("optimizer", "AdamW")
        self.save_animations = config.get("_save_animations", True)
        self.loss_function = CustomLoss(config.get("loss_weights"))
        self.load = config.get("load", False)
        self.checkpoint_path = config.get("_checkpoint_path", None)
        self.activation = config.get("activation", "relu")
        self.activation = activation_dict[self.activation]
        self.transformer_activation = config.get("transformer_activation", "gelu")
        self.output_layer = config.get("output_layer", "linear")
        self.clip = config.get("clip_grad_norm", 1)
        self.batch_norm = config.get("batch_norm", False)
        self.hindden_encoder_layer_widths = config.get("hidden_encoder_layer_widths", [256] * 3 )
        self.hidden_dim_trans = config.get("hidden_dim_trans", 8192)

        

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.input_dim,
                nhead=self.n_heads,
                dim_feedforward=self.hidden_dim_trans,
                dropout=self.dropout,
                batch_first=True,
                activation=self.transformer_activation,
                norm_first=True,
            ),
            num_layers=self.n_layers,
            norm=nn.LayerNorm(self.input_dim),
        )

        self.encoder_linear_block = nn.Sequential(
            nn.Linear(self.input_dim * self.seq_len, 8 * self.hidden_dim),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim*8, self.hidden_dim*4),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim *4, self.hidden_dim*2),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, 2 * self.latent_dim),
        )

        self.decoder = Decoder(
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            seq_len=self.seq_len,
            input_dim=self.input_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dropout=self.dropout,
            hidden_dim_trans=self.hidden_dim_trans,
            transformer_activation=self.transformer_activation,
            activation=self.activation,
        )

        if self.load:
            print(f"Loading model from {self.checkpoint_path}")
            weights = torch.load(self.checkpoint_path)
            # enc_weight_keys = list(weights['state_dict'].keys())[:70]
            # dec_weight_keys = list(weights['state_dict'].keys())[70:]
            # self.load_state_dict({k: v for k, v in weights['state_dict'].items() if k in enc_weight_keys}, strict=False)
            # self.decoder.load_state_dict({k: v for k, v in weights['state_dict'].items() if k in dec_weight_keys}, strict=False)
            self.load_state_dict(weights['state_dict'])
            print('loaded model from:', self.checkpoint_path)

        

        self.epochs_animated = []

    def encode(self, x):
        x = x.view(-1, self.seq_len, self.input_dim)
        x = self.transformer_encoder(x)
        x = x.view(-1, self.input_dim * self.seq_len)
        x = self.encoder_linear_block(x)
        mu, logvar = x.chunk(2, dim=1)
        return mu, logvar
    
    def decode(self, z):
        return self.decoder(z)


    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x = self.decode(z)
        return x, mu, logvar, z

    def training_step(self, batch, batch_idx):
        res = self._common_step(batch, batch_idx)

        
        loss = {k + '_trn': v for k, v in res['loss'].items()}
        self.log_dict(loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        
        # clip gradients --> do i do this here? # TODO
        if self.clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
        return res['loss']['total']

    def validation_step(self, batch, batch_idx):
        res = self._common_step(batch, batch_idx)
        loss = {k + '_val': v for k, v in res['loss'].items()}
        self.log_dict(loss)
        current_epoch = self.current_epoch
        if current_epoch not in self.epochs_animated:
            self.epochs_animated.append(current_epoch)
            print()
            recon = res['recon']
            x = res['motion_seq']
            im_arr = plot_3d_motion_frames_multiple([recon[0].cpu().detach().numpy(), x[0].cpu().detach().numpy()], ["recon", "true"], 
                                                    nframes=5, radius=2, figsize=(20,8), return_array=True, velocity=False)
            # print(im_arr.shape)
            self.logger.experiment.add_image("recon_vs_true", im_arr, global_step=self.global_step)
            if self.save_animations:
                
                # print("Saving animations")
                folder = self.logger.log_dir
                fname = f"{folder}/recon_epoch{current_epoch}.mp4"
                plot_3d_motion_animation(recon[1].cpu().detach().numpy(), "recon", 
                                        figsize=(10, 10), fps=20, radius=2, save_path=fname, velocity=False)
                plt.close()

                # copy file to latest
                import shutil
                shutil.copyfile(fname, f"{folder}/recon_latest.mp4")


                if current_epoch == 0:
                    plot_3d_motion_animation(x[1].cpu().detach().numpy(), "true", 
                                            figsize=(10, 10), fps=20, radius=2, save_path=f"{folder}/recon_true.mp4", velocity=False)
                    plt.close()

    def test_step(self, batch, batch_idx):
        res = self._common_step(batch, batch_idx)
        loss = {k + '_tst': v for k, v in res['loss'].items()}
        #self.log("test_loss", loss)
        # we want to add test loss final to the tensorboard
        self.log_dict(loss)

        if batch_idx == 1 and self.save_animations:
            recon = res['recon']
            print("Saving animations")
            folder = self.logger.log_dir
            plot_3d_motion_animation(recon[0].cpu().detach().numpy(), "recon", 
                                     figsize=(10, 10), fps=20, radius=2, save_path=f"{folder}/recon_test.mp4", velocity=False)
            plt.close()
        return loss
    
    def decompose_recon(self, motion_seq):
        pose0 = motion_seq[:,:1]
        root_travel = motion_seq[:, :, :1, :]
        root_travel = root_travel - root_travel[:1]  # relative to the first frame
        motion_less_root = motion_seq - root_travel# relative motion
        velocity = torch.diff(motion_seq, dim=1)
        velocity_relative = torch.diff(motion_less_root, dim=1)

        return pose0, velocity_relative, root_travel



    def _common_step(self, batch, batch_idx, verbose=False):
        pose0,  velocity_relative, root_travel, motion_seq, text = batch
        motion_less_root = motion_seq - root_travel# relative motion

        recon, mu, logvar, z = self(motion_seq)
        pose0_rec, vel_rec, root_rec = self.decompose_recon(recon)
        motion_less_root_rec = recon - root_rec
        if verbose:
            print('motion:', motion_seq.shape)
            print('pose0:', pose0.shape)
            print('velocity_relative:', velocity_relative.shape)
            print('root:', root_travel.shape)
            print('recon:', recon.shape)
            print('pose0_rec:', pose0_rec.shape)
            print('vel_rec:', vel_rec.shape)
            print('root_rec:', root_rec.shape)

        loss_data = {
            'velocity_relative' : {'true': velocity_relative, 'rec': vel_rec, },#'weight': 50},      
            'root' : {'true': root_travel, 'rec': root_rec,},# 'weight':          1},      
            'pose0' : {'true': pose0, 'rec': pose0_rec, },#'weight':              1},
            'motion' : {'true': motion_seq, 'rec': recon,},# 'weight':            0.},
            'motion_relative' : {'true': motion_less_root, 
                                 'rec': motion_less_root_rec, },#'weight':       100},
        }
        # print('loss_data:', {k : v for k, v in loss_data.items() if v['weight'] > 0})


        loss = self.loss_function(loss_data, mu, logvar)
        # loss  = {'total' : F.mse_loss(recon, motion_seq)}
        return dict(
            loss=loss,
            motion_seq=motion_seq,
            recon=recon,
            pose0={'true': pose0, 'rec': pose0_rec},
            vel = {'true': velocity_relative, 'rec': vel_rec},
            root = {'true': root_travel, 'rec': root_rec},
            mu=mu,
            logvar=logvar,
            text=text,
        )

    def configure_optimizers(self):
        # this is also where we would put the scheduler
        return get_optimizer(self, self.optimizer, self.lr)


if __name__ == "__main__":
    model = TransformerMotionAutoencoder(
        seq_len=160,
        input_dim=66,
        hidden_dim=128,
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
    x = torch.randn(128, 160, 66)
    recon, mu, logvar = model(x)
    print(recon.shape)