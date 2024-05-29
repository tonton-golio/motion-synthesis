import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils_pose import (
    plot_3d_motion_frames_multiple,
    plot_3d_motion_animation,
    plot_3d_motion_frames_multiple,
)
from glob import glob
import matplotlib.pyplot as plt

activation_dict = {
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "elu": nn.ELU(),
    "swish": nn.SiLU(),
    "mish": nn.Mish(),
    "softplus": nn.Softplus(),
    "softsign": nn.Softsign(),
    # 'bent_identity': nn.BentIdentity(),
    # 'gaussian': nn.Gaussian(),
    "softmax": nn.Softmax(),
    "softmin": nn.Softmin(),
    "softshrink": nn.Softshrink(),
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
            loss[key] = F.mse_loss(data["rec"], data["true"]) * self.loss_weights[key]
            # loss[key] = F.l1_loss(data['rec'], data['true']) * self.loss_weights[key]
            total_loss += loss[key]

        kl_loss = self.kl_divergence(mu, logvar) * self.loss_weights["kl_div"]
        total_loss += kl_loss
        loss["kl_divergence"] = kl_loss
        loss["total"] = total_loss
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


class Model(nn.Module):
    def __init__(self, latent_dim=512, seq_len=120):
        super().__init__()
        self.seq_len = seq_len

        # Encoder
        self.transformer_motion_enc = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=66,
                nhead=6,
                dim_feedforward=2048,
                dropout=0.1,
                activation="relu",
            ),
            num_layers=6,
        )

        self.motion_linear = nn.Linear(66, 31)

        self.text_embedding = nn.Embedding(100194 + 1, 128)
        self.transformer_text = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                dropout=0.1,
                activation="relu",
            ),
            num_layers=3,
        )
        self.text_linear = nn.Linear(128, 16)
        self.text_linear2 = nn.Linear(1600, 120)

        # self.enc_linear = nn.Linear(1920, latent_dim*2)
        self.enc_linear = nn.Sequential(
            nn.Linear(3840, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, latent_dim * 2),
        )

        # Decoder
        self.dec_linear = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, seq_len * 66),
        )

        self.decoder_transformer = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(
                d_model=66,
                nhead=6,
                dim_feedforward=2048,
                dropout=0.1,
                activation="relu",
            ),
            num_layers=6,
        )

        self.linear_out = nn.Linear(66, 66)

    def encode(self, motion, text_enc, verbose=True):
        # batch norm
        # motion = self.batch_norm(motion)

        motion = motion.view(motion.shape[0], motion.shape[1], -1)
        if verbose:
            print("motion:", motion.shape)
        motion = self.transformer_motion_enc(motion)
        if verbose:
            print("motion:", motion.shape)
        motion = self.motion_linear(motion)

        if verbose:
            print("motion:", motion.shape)

        text_enc = self.text_embedding(text_enc)
        if verbose:
            print("text_enc:", text_enc.shape)
        text = self.transformer_text(text_enc)
        if verbose:
            print("text:", text.shape)
        text = self.text_linear(text)
        text = nn.ReLU()(text)
        if verbose:
            print("text:", text.shape)
        text = nn.Flatten()(text)
        if verbose:
            print("text:", text.shape)
        text = self.text_linear2(text).view(motion.shape[0], motion.shape[1], -1)
        text = nn.ReLU()(text)
        if verbose:
            print("text:", text.shape)

        # print(motion.shape, text.shape)

        x = torch.cat([motion, text], dim=2)
        if verbose:
            print("x:", x.shape)
        x = nn.Flatten()(x)
        if verbose:
            print("x:", x.shape)
        x = self.enc_linear(x)
        if verbose:
            print("x:", x.shape)
        mu, logvar = x.chunk(2, dim=1)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, verbose=True):
        if verbose:
            print("x:", x.shape)
        x = self.dec_linear(x)
        if verbose:
            print("x:", x.shape)
        x = x.view(x.shape[0], self.seq_len, 66)
        if verbose:
            print("x:", x.shape)
        # print(x.shape)
        x = self.decoder_transformer(x, x)
        if verbose:
            print("x:", x.shape)
        x = self.linear_out(x)
        return x.view(x.shape[0], self.seq_len, 22, 3)

    def forward(self, motion, text_enc, verbose=False):
        mu, logvar = self.encode(motion, text_enc, verbose)
        z = self.reparametrize(mu, logvar)
        x = self.decode(z, verbose)
        return x


class TransformerMotionAutoencoder(pl.LightningModule):
    def __init__(
        self,
        **config,
    ):
        super(TransformerMotionAutoencoder, self).__init__()
        self.latent_dim = config.get("latent_dim", 512)
        self.seq_len = config.get("seq_len", 120)
        self.lr = config.get("learning_rate", 1e-5)
        self.loss_function = CustomLoss(config.get("loss_weights"))
        self.model = Model(latent_dim=self.latent_dim, seq_len=self.seq_len)
        self.save_animations = config.get("_save_animations", True)
        self.epochs_animated = []
        self.clip = config.get("clip_grad_norm", 0)
        self.checkpoint_path = config.get("_checkpoint_path", "latest")
        self.batch_size = config.get("batch_size", 128)
        # load
        if config.get("load", False):
            print(f"Loading model from {self.checkpoint_path}")
            weights = torch.load(self.checkpoint_path)
            self.load_state_dict(weights["state_dict"])
            print("loaded model from:", self.checkpoint_path)

    def forward(self, x, text_enc, verbose=True):
        mu, logvar = self.model.encode(x, text_enc, verbose)

        z = self.model.reparametrize(mu, logvar)
        x = self.model.decode(z, verbose)
        return x, mu, logvar, z

    def training_step(self, batch, batch_idx):
        res = self._common_step(batch, batch_idx)

        loss = {k + "_trn": v for k, v in res["loss"].items()}
        self.log_dict(
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
            batch_size=self.batch_size,
        )

        # clip gradients --> do i do this here? # TODO
        if self.clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
        return res["loss"]["total"]

    def validation_step(self, batch, batch_idx):
        res = self._common_step(batch, batch_idx)
        loss = {k + "_val": v for k, v in res["loss"].items()}
        self.log_dict(loss, batch_size=self.batch_size)
        current_epoch = self.current_epoch
        if current_epoch not in self.epochs_animated:
            self.epochs_animated.append(current_epoch)
            print()
            recon = res["recon"]
            text = res["text"]
            x = res["motion_seq"]
            im_arr = plot_3d_motion_frames_multiple(
                [recon[0].cpu().detach().numpy(), x[0].cpu().detach().numpy()],
                ["recon", "true"],
                nframes=5,
                radius=2,
                figsize=(20, 8),
                return_array=True,
                velocity=False,
            )
            # print(im_arr.shape)
            self.logger.experiment.add_image(
                "recon_vs_true", im_arr, global_step=self.global_step
            )
            if self.save_animations:
                # print("Saving animations")
                folder = self.logger.log_dir
                fname = f"{folder}/recon_epoch{current_epoch}.mp4"
                plot_3d_motion_animation(
                    recon[0].cpu().detach().numpy(),
                    text[0],
                    figsize=(10, 10),
                    fps=20,
                    radius=2,
                    save_path=fname,
                    velocity=False,
                )
                plt.close()

                # copy file to latest
                import shutil

                shutil.copyfile(fname, f"{folder}/recon_latest.mp4")

                if current_epoch == 0:
                    plot_3d_motion_animation(
                        x[0].cpu().detach().numpy(),
                        "true",
                        figsize=(10, 10),
                        fps=20,
                        radius=2,
                        save_path=f"{folder}/recon_true.mp4",
                        velocity=False,
                    )
                    plt.close()

    def test_step(self, batch, batch_idx):
        res = self._common_step(batch, batch_idx)
        loss = {k + "_tst": v for k, v in res["loss"].items()}
        # self.log("test_loss", loss)
        # we want to add test loss final to the tensorboard
        self.log_dict(loss, batch_size=self.batch_size)

        if batch_idx == 1 and self.save_animations:
            recon = res["recon"]
            text = res["text"]
            print("Saving animations")
            folder = self.logger.log_dir
            plot_3d_motion_animation(
                recon[0].cpu().detach().numpy(),
                text[0],
                figsize=(10, 10),
                fps=20,
                radius=2,
                save_path=f"{folder}/recon_test.mp4",
                velocity=False,
            )
            plt.close()
        return loss

    def decompose_recon(self, motion_seq):
        pose0 = motion_seq[:, :1]
        root_travel = motion_seq[:, :, :1, :]
        root_mag = torch.norm(root_travel, dim=3, keepdim=True)
        # root_travel = root_travel - root_travel[:1]  # relative to the first frame
        motion_less_root = motion_seq - root_travel  # relative motion
        velocity = torch.diff(motion_seq, dim=1) * 20  # fps = 20, [m/s]
        velocity_relative = torch.diff(motion_less_root, dim=1) * 20  # fps = 20, [m/s]

        return velocity, motion_less_root, root_mag, velocity_relative

    def _common_step(self, batch, batch_idx, verbose=False):
        motion, velocity, text_enc, text = batch

        recon, mu, logvar, z = self(motion, text_enc, verbose)
        # print('recon:', recon.shape)
        (
            vel_recon,
            motion_less_root_recon,
            root_mag_recon,
            velocity_relative_recon,
        ) = self.decompose_recon(recon)
        (
            vel_true,
            motion_less_root_true,
            root_mag,
            velocity_relative_true,
        ) = self.decompose_recon(motion)

        loss_data = {
            "velocity": {
                "true": vel_true,
                "rec": vel_recon,
            },
            "motion": {
                "true": motion,
                "rec": recon,
            },
            "motion_relative": {
                "true": motion_less_root_true,
                "rec": motion_less_root_recon,
            },
            "root": {
                "true": root_mag,
                "rec": root_mag_recon,
            },
            "velocity_relative": {
                "true": velocity_relative_true,
                "rec": velocity_relative_recon,
            },
        }

        # if self.current_epoch % 3 == 0: # vel
        #     loss_data = {'velocity' : {'true': velocity, 'rec': vel_recon}}
        # elif self.current_epoch % 3 == 1: #motion
        #     loss_data = {'motion' : {'true': motion, 'rec': recon}}
        # else: # motion relative
        #     loss_data = {'motion_relative' : {'true': motion_less_root_true, 'rec': motion_less_root_recon}}

        loss = self.loss_function(loss_data, mu, logvar)
        # loss  = {'total' : F.mse_loss(recon, motion_seq)}
        return dict(
            loss=loss,
            motion_seq=motion,
            recon=recon,
            text=text,
        )

    def configure_optimizers(self):
        # this is also where we would put the scheduler
        return optim.Adam(self.parameters(), lr=self.lr)
