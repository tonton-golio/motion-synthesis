import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import (
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


class CustomLoss(nn.Module):
    def __init__(self, klDiv_weight=0.00000002, relative_weight=30, root_weight=1):
        super(CustomLoss, self).__init__()
        self.klDiv_weight = klDiv_weight
        self.relative_weight = relative_weight
        self.root_weight = root_weight

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
            loss[key] = F.mse_loss(data["rec"], data["true"]) * data["weight"]
            total_loss += loss[key]
        kl_loss = self.kl_divergence(mu, logvar) * self.klDiv_weight
        total_loss += kl_loss
        loss["kl_divergence"] = kl_loss
        loss["total"] = total_loss

        return loss

    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


# no activation class
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class TransformerMotionAutoencoder_Chunked(pl.LightningModule):
    """
    We want this class to recieve:
        pose0: (batch_size, 1, num_joints=22, 3)
        velocity_relative: (batch_size, seq_len-1, num_joints=22, 3)
        root_travel: (batch_size, seq_len, 1, 3)
    from the dataloader and output:
        the same pose0,  velocity_relative, root_travel
    """

    def __init__(
        self,
        **cfg,
    ):
        super(TransformerMotionAutoencoder_Chunked, self).__init__()

        # data things
        self.seq_len = cfg.get("seq_len", 160)
        self.input_dim = cfg.get("input_dim", 66)

        # model things
        self.hidden_dim = cfg.get("hidden_dim", 1024)
        self.n_layers = cfg.get("n_layers", 8)
        self.n_heads = cfg.get("n_heads", 6)
        self.dropout = cfg.get("dropout", 0.10)
        self.latent_dim = cfg.get("latent_dim", 256)
        self.activation = cfg.get("activation", "relu")
        self.activation = activation_dict[self.activation]
        self.transformer_activation = cfg.get("transformer_activation", "gelu")
        self.output_layer = cfg.get("output_layer", "linear")
        self.clip = cfg.get("clip_grad_norm", 1)
        self.batch_norm = cfg.get("batch_norm", False)
        self.hindden_encoder_layer_widths = cfg.get(
            "hidden_encoder_layer_widths", [256] * 3
        )

        # training things
        self.lr = cfg.get("learning_rate", 1 * 1e-5)
        self.optimizer = cfg.get("optimizer", "AdamW")
        self.load = cfg.get("load", False)
        self.checkpoint_path = cfg.get("_checkpoint_path", None)

        # logging things
        self.save_animations = cfg.get("_save_animations", True)
        self.loss_function = CustomLoss(cfg.get("klDiv_weight", 0.000001))

        ##### MODEL #####
        # the strucure will be as follows:
        # 0. con
        ## 1. encoder
        ## chunked transformer
        ## concat
        ## linear layer
        ## reparameterize
        ## 2. decoder
        ## linear layer
        ## chunked transformer
        ## concat

        # vel transformer encoder
        cfg_vte = cfg['vel_transformer_encoder']
        cfg['vel_transformer_encoder']= nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=66,
                nhead=6,  # cfg_vte.get("nhead", 6),
                dim_feedforward=2048,  # cfg_vte.get("dim_feedforward", 1024),
                dropout=0.1,  # cfg_vte.get("dropout", 0.1),
                activation=cfg_vte.get("activation", "gelu"),
            ),
            num_layers=6,  # cfg_vte.get("num_layers", 6),
        )

        # vel linear encoder
        cfg_vle = cfg['vel_linear_encoder']
        self.vel_linear_encoder = nn.Sequential()
        current_dim = (self.seq_len - 1) * 66
        in_dims = [current_dim] + cfg_vle.get("hidden_layer_widths", [4096] * 3)
        out_dims = in_dims[1:] + [16 * self.latent_dim]
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            self.vel_linear_encoder.add_module(
                f"linear_{i}", nn.Linear(in_dim, out_dim)
            )
            # if i< len(in_dims) - 1:
            self.vel_linear_encoder.add_module(
                f"activation_{i}", activation_dict[cfg_vle.get("activation", "relu")]
            )
            if self.batch_norm:
                self.vel_linear_encoder.add_module(
                    f"batch_norm_{i}", nn.BatchNorm1d(out_dim)
                )
            self.vel_linear_encoder.add_module(
                f"dropout_{i}", nn.Dropout(cfg_vle.get("dropout", 0.1))
            )

        self.vel_encoded_dim = out_dim

        # root transformer encoder
        cfg_rte = cfg['root_travel_transformer_encoder']
        self.root_travel_transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=3,
                nhead=cfg_rte.get("nhead", 3),
                dim_feedforward=cfg_rte.get("dim_feedforward", 1024),
                dropout=cfg_rte.get("dropout", 0.1),
                activation=cfg_rte.get("activation", "relu"),
            ),
            num_layers=cfg_rte.get("num_layers", 6),
        )

        # root linear encoder
        cfg_rle = cfg['root_linear_encoder']
        self.root_linear_encoder = nn.Sequential()
        current_dim = 3 * self.seq_len
        in_dims = [current_dim] + cfg_rle.get("hidden_layer_widths", [256] * 3)
        out_dims = in_dims[1:] + [2 * self.latent_dim]
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            self.root_linear_encoder.add_module(
                f"linear_{i}", nn.Linear(in_dim, out_dim)
            )
            # if i< len(in_dims) - 1:
            self.root_linear_encoder.add_module(
                f"activation_{i}", activation_dict[cfg_rle.get("activation", "relu")]
            )
            if self.batch_norm:
                self.root_linear_encoder.add_module(
                    f"batch_norm_{i}", nn.BatchNorm1d(out_dim)
                )
            self.root_linear_encoder.add_module(
                f"dropout_{i}", nn.Dropout(cfg_rle.get("dropout", 0.1))
            )

        self.root_encoded_dim = out_dim

        # linear encoder
        cfg_le = cfg['linear_encoder']
        self.linear_encoder = nn.Sequential()
        current_dim = 18 * self.latent_dim + 66
        in_dims = [current_dim] + cfg_le.get("hidden_encoder_layer_widths", [1024] * 3)
        out_dims = in_dims[1:] + [2 * self.latent_dim]
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            self.linear_encoder.add_module(f"linear_{i}", nn.Linear(in_dim, out_dim))
            if i < len(in_dims) - 1:
                self.linear_encoder.add_module(
                    f"activation_{i}", activation_dict[cfg_le.get("activation", "relu")]
                )
                if self.batch_norm:
                    self.linear_encoder.add_module(
                        f"batch_norm_{i}", nn.BatchNorm1d(out_dim)
                    )
                self.linear_encoder.add_module(
                    f"dropout_{i}", nn.Dropout(cfg_le.get("dropout", 0.1))
                )

        # linear decoder
        cfg_ld = cfg['linear_decoder']
        self.linear_decoder = nn.Sequential()
        current_dim = self.latent_dim
        target_dim = 18 * self.latent_dim + 66
        in_dims = [current_dim] + cfg_ld.get("hidden_decoder_layer_widths", [1024] * 3)
        out_dims = cfg_ld.get("hidden_decoder_layer_widths", [1024] * 3) + [target_dim]
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            self.linear_decoder.add_module(f"linear_{i}", nn.Linear(in_dim, out_dim))
            # if i>0:
            self.linear_decoder.add_module(
                f"activation_{i}", activation_dict[cfg_ld.get("activation", "relu")]
            )
            if self.batch_norm:
                self.linear_decoder.add_module(
                    f"batch_norm_{i}", nn.BatchNorm1d(out_dim)
                )
            self.linear_decoder.add_module(
                f"dropout_{i}", nn.Dropout(cfg_ld.get("dropout", 0.1))
            )

        # vel linear decoder
        cfg_vld = cfg['vel_linear_decoder']
        self.vel_linear_decoder = nn.Sequential()
        target_dim = (self.seq_len - 1) * 66
        current_dim = self.vel_encoded_dim
        in_dims = [current_dim] + cfg_vld.get("hidden_decoder_layer_widths", [1024] * 3)
        out_dims = cfg_vld.get("hidden_decoder_layer_widths", [1024] * 3) + [target_dim]
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            self.vel_linear_decoder.add_module(
                f"linear_{i}", nn.Linear(in_dim, out_dim)
            )
            # if i>0:
            self.vel_linear_decoder.add_module(
                f"activation_{i}", activation_dict[cfg_vld.get("activation", "relu")]
            )
            if self.batch_norm:
                self.vel_linear_decoder.add_module(
                    f"batch_norm_{i}", nn.BatchNorm1d(out_dim)
                )
            self.vel_linear_decoder.add_module(
                f"dropout_{i}", nn.Dropout(cfg_vld.get("dropout", 0.1))
            )

        # vel transformer decoder
        cfg_vtd = cfg['vel_transformer_decoder']
        self.vel_transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=66,
                nhead=cfg_vtd.get("nhead", 6),
                dim_feedforward=cfg_vtd.get("dim_feedforward", 1024),
                dropout=cfg_vtd.get("dropout", 0.1),
                activation=cfg_vtd.get("activation", "relu"),
            ),
            num_layers=cfg_vtd.get("num_layers", 6),
        )

        # root_linear_decoder
        cfg_rld = cfg['root_linear_decoder']
        self.root_linear_decoder = nn.Sequential()
        target_dim = 3 * self.seq_len
        current_dim = self.root_encoded_dim
        in_dims = [current_dim] + cfg_rld.get("hidden_decoder_layer_widths", [1024] * 3)
        out_dims = cfg_rld.get("hidden_decoder_layer_widths", [1024] * 3) + [target_dim]
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            self.root_linear_decoder.add_module(
                f"linear_{i}", nn.Linear(in_dim, out_dim)
            )
            # if i>0:
            self.root_linear_decoder.add_module(
                f"activation_{i}", activation_dict[cfg_rld.get("activation", "relu")]
            )
            if self.batch_norm:
                self.root_linear_decoder.add_module(
                    f"batch_norm_{i}", nn.BatchNorm1d(out_dim)
                )
            self.root_linear_decoder.add_module(
                f"dropout_{i}", nn.Dropout(cfg_rld.get("dropout", 0.1))
            )

        # root transformer decoder
        cfg_rtd = cfg['root_travel_transformer_decoder']
        self.root_travel_transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=3,
                nhead=cfg_rtd.get("nhead", 3),
                dim_feedforward=cfg_rtd.get("dim_feedforward", 512),
                dropout=cfg_rtd.get("dropout", 0.1),
                activation=cfg_rtd.get("activation", "relu"),
            ),
            num_layers=cfg_rtd.get("num_layers", 6),
        )

    def encode(self, pose0, velocity_relative, root_travel, verbose=False):
        if verbose:
            print("\nStarting encoder")
            print("initial shapes:")
            print("     pose0:", pose0.shape)
            print("     velocity_relative:", velocity_relative.shape)
            print("     root_travel:", root_travel.shape)
        # pose0: (batch_size, 1, num_joints=22, 3)
        # velocity_relative: (batch_size, seq_len-1, num_joints=22, 3)
        # root_travel: (batch_size, seq_len, 1, 3)

        # 1. encoder
        # chunked transformer for vel and roof
        # concat
        # linear layer
        pose0 = pose0.view(-1, 66)
        velocity_relative = velocity_relative.view(-1, self.seq_len - 1, 66)
        root_travel = root_travel.view(-1, self.seq_len, 3)
        if verbose:
            print("pose0:", pose0.shape)
            print("velocity_relative:", velocity_relative.shape)
            print("root_travel:", root_travel.shape)
        # vel
        vel = self.vel_transformer_encoder(
            velocity_relative,
        )
        if verbose:
            print("vel:", vel.shape)
        vel = nn.Flatten()(vel)
        if verbose:
            print("vel:", vel.shape)
        vel = self.vel_linear_encoder(vel)
        if verbose:
            print("vel:", vel.shape)

        # root
        root = self.root_travel_transformer_encoder(
            root_travel,
        )
        if verbose:
            print("root:", root.shape)
        root = nn.Flatten()(root)
        if verbose:
            print("root:", root.shape)
        root = self.root_linear_encoder(root)
        if verbose:
            print("root:", root.shape)

        # concat
        x = torch.cat([nn.Flatten()(vel), nn.Flatten()(root), pose0], dim=1)
        if verbose:
            print("concat:", x.shape)
        x = self.linear_encoder(x)
        if verbose:
            print("linear_encoder:", x.shape)
        mu, logvar = torch.chunk(x, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu.__add__(eps.__mul__(std))

    def decode(self, z, verbose=False):
        if verbose:
            print("\nStarting decoder")

        # linear layer
        # chunked transformer
        # concat
        if verbose:
            print("z:", z.shape)
        z = self.linear_decoder(z)
        if verbose:
            print("z:", z.shape)
        # vel
        # print(z.shape)
        if verbose:
            print("self.vel_encoded_dim:", self.vel_encoded_dim)
        if verbose:
            print("self.root_encoded_dim:", self.root_encoded_dim)
        vel = z[:, : self.vel_encoded_dim]
        root = z[:, self.vel_encoded_dim : self.vel_encoded_dim + self.root_encoded_dim]
        pose0 = z[:, self.vel_encoded_dim + self.root_encoded_dim :]
        if verbose:
            print("pose0:", pose0.shape)
            print("vel:", vel.shape)
            print("root:", root.shape)
        vel = self.vel_linear_decoder(vel)
        if verbose:
            print("vel:", vel.shape)
        vel = vel.view(-1, self.seq_len - 1, 66)
        vel = self.vel_transformer_decoder(vel, vel)
        if verbose:
            print("vel:", vel.shape)

        root = self.root_linear_decoder(root)
        if verbose:
            print("root:", root.shape)
        root = root.view(-1, self.seq_len, 3)

        root = self.root_travel_transformer_decoder(root, root)
        if verbose:
            print("root:", root.shape)
        vel = vel.view(-1, self.seq_len - 1, 22, 3)
        root = root.view(-1, self.seq_len, 1, 3)
        pose0 = pose0.view(-1, 1, 22, 3)

        if verbose:
            print("final shapes:")
            print("     pose0:", pose0.shape)
            print("     vel:", vel.shape)
            print("     root:", root.shape)

        return pose0, vel, root

    def forward(self, pose0, velocity_relative, root_travel):
        mu, logvar = self.encode(pose0, velocity_relative, root_travel)
        z = self.reparameterize(mu, logvar)
        pose0_rec, vel_rec, root_rec = self.decode(z)
        # return None
        # return pose0_rec, vel_rec, root_rec, mu, logvar
        # relative_motion = self.reconstruct(pose0_rec, vel_rec)
        return pose0_rec, vel_rec, root_rec, mu, logvar

    def reconstruct(self, pose0, velocity_relative, root_travel=None, batch=False):
        # print('pose0:', pose0.shape)
        # print('velocity_relative:', velocity_relative.shape)
        motion_less_root = torch.cumsum(
            torch.cat([pose0, velocity_relative], dim=1 if batch else 0),
            dim=1 if batch else 0,
        )
        if root_travel is not None:
            # print('root_travel:', root_travel.shape)
            return motion_less_root + root_travel
        return motion_less_root

    def reconstruction_step(self, res):
        idx = torch.randint(0, res["pose0"]["true"].shape[0], (1,)).item()
        recon = self.reconstruct(
            res["pose0"]["rec"][idx], res["vel"]["rec"][idx], res["root"]["rec"][idx]
        )
        recon = recon.cpu().detach().numpy()

        true = self.reconstruct(
            res["pose0"]["true"][idx], res["vel"]["true"][idx], res["root"]["true"][idx]
        )
        true = true.cpu().detach().numpy()
        return recon, true

    def training_step(self, batch, batch_idx):
        res = self._common_step(batch, batch_idx)

        loss = {k + "_trn": v for k, v in res["loss"].items()}
        self.log_dict(loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        # clip gradients --> do i do this here? # TODO
        # if self.clip > 0:
        #     torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
        return res["loss"]["total"]

    def validation_step(self, batch, batch_idx):
        res = self._common_step(batch, batch_idx)
        loss = {k + "_val": v for k, v in res["loss"].items()}
        self.log_dict(loss)

        if batch_idx == 0:
            print()
            recon, true = self.reconstruction_step(res)
            im_arr = plot_3d_motion_frames_multiple(
                [recon, true],
                ["recon", "true"],
                nframes=5,
                radius=2,
                figsize=(20, 8),
                return_array=True,
                velocity=False,
            )

            self.logger.experiment.add_image(
                "recon_vs_true", im_arr, global_step=self.global_step
            )
            if self.save_animations:
                # print("Saving animations")
                folder = self.logger.log_dir
                plot_3d_motion_animation(
                    recon,
                    "recon",
                    figsize=(10, 10),
                    fps=20,
                    radius=2,
                    save_path=f"{folder}/recon.mp4",
                    velocity=False,
                )
                plt.close()

    def test_step(self, batch, batch_idx):
        res = self._common_step(batch, batch_idx)
        loss = {k + "_tst": v for k, v in res["loss"].items()}
        # self.log("test_loss", loss)
        # we want to add test loss final to the tensorboard
        self.log_dict(loss)
        idx = 0
        if batch_idx == idx and self.save_animations:
            recon, true = self.reconstruction_step(res)
            print("Saving animations")
            folder = self.logger.log_dir
            plot_3d_motion_animation(
                recon.cpu().detach().numpy(),
                "recon",
                figsize=(10, 10),
                fps=20,
                radius=2,
                save_path=f"{folder}/recon.mp4",
                velocity=False,
            )
            plt.close()
        return loss

    def _common_step(self, batch, batch_idx):
        pose0, vel, root = batch
        motion = self.reconstruct(pose0, vel, root, batch=True)
        pose0_rec, vel_rec, root_rec, mu, logvar = self(pose0, vel, root)
        motion_rec = self.reconstruct(pose0, vel_rec, root_rec, batch=True)
        loss_data = {
            "velocity": {"true": vel, "rec": vel_rec, "weight": 1},
            "root": {"true": root, "rec": root_rec, "weight": 1},
            "pose0": {"true": pose0, "rec": pose0_rec, "weight": 1},
        }
        loss = self.loss_function(loss_data, mu, logvar)
        # loss  = {'total' : F.mse_loss(motion, motion_rec)}
        return dict(
            loss=loss,
            pose0={"true": pose0, "rec": pose0_rec},
            vel={"true": vel, "rec": vel_rec},
            root={"true": root, "rec": root_rec},
            mu=mu,
            logvar=logvar,
        )

    def configure_optimizers(self):
        # this is also where we would put the scheduler
        return get_optimizer(self, self.optimizer, self.lr)


class TransformerMotionAutoencoder_Concatenated(pl.LightningModule):
    """
    We want this class to recieve:
        pose0: (batch_size, 1, num_joints=22, 3)
        velocity_relative: (batch_size, seq_len-1, num_joints=22, 3)
        root_travel: (batch_size, seq_len, 1, 3)
    from the dataloader and output:
        the same pose0,  velocity_relative, root_travel
    """

    def __init__(
        self,
        cfg,
    ):
        super(TransformerMotionAutoencoder_Concatenated, self).__init__()
        self.cfg = cfg
        # data things
        self.seq_len = cfg.get("seq_len", 160)
        self.input_dim = cfg.get("input_dim", 66)

        # model things
        self.hidden_dim = cfg.get("hidden_dim", 1024)
        self.n_layers = cfg.get("n_layers", 8)
        self.n_heads = cfg.get("n_heads", 6)
        self.dropout = cfg.get("dropout", 0.10)
        self.latent_dim = cfg.get("latent_dim", 256)
        self.activation = cfg.get("activation", "relu")
        self.activation = activation_dict[self.activation]
        self.transformer_activation = cfg.get("transformer_activation", "gelu")
        self.output_layer = cfg.get("output_layer", "linear")
        self.clip = cfg.get("clip_grad_norm", 1)
        self.batch_norm = cfg.get("batch_norm", False)
        self.hindden_encoder_layer_widths = cfg.get(
            "hidden_encoder_layer_widths", [256] * 3
        )

        # training things
        self.lr = cfg.get("learning_rate", 1 * 1e-5)
        self.optimizer = cfg.get("optimizer", "AdamW")
        self.load = cfg.get("load", False)
        self.checkpoint_path = cfg.get("_checkpoint_path", None)

        # logging things
        self.save_animations = cfg.get("_save_animations", True)
        self.loss_function = CustomLoss(cfg.get("klDiv_weight", 0.000001))

        ##### MODEL #####
        # the strucure will be as follows:
        # 0. concat
        ## 1. encoder
        ## linear
        ##  transformer
        ## flatten
        ## linear layer
        ## reparameterize
        ## 2. decoder
        ## linear layer
        ## chunked transformer
        ## concat
        self.cfg.CONCAT_TRANSFORMER.linear_encoder_input.out_dim = 100
        cfg_lei = self.cfg.CONCAT_TRANSFORMER.linear_encoder_input

        # print(cfg_lei)
        self.linear_encoder_input = nn.Sequential()
        current_dim = (self.seq_len - 1) * 66 + 3 * self.seq_len + 66
        in_dims = [current_dim] + cfg_lei.get(
            "hidden_encoder_layer_widths", [current_dim]
        )
        out_dims = in_dims[1:] + [self.seq_len * cfg_lei.get("out_dim", 128)]
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            self.linear_encoder_input.add_module(
                f"linear_{i}", nn.Linear(in_dim, out_dim)
            )
            self.linear_encoder_input.add_module(
                f"activation_{i}", activation_dict[cfg_lei.get("activation", "relu")]
            )
            if self.batch_norm:
                self.linear_encoder_input.add_module(
                    f"batch_norm_{i}", nn.BatchNorm1d(out_dim)
                )
            self.linear_encoder_input.add_module(
                f"dropout_{i}", nn.Dropout(cfg_lei.get("dropout", 0.1))
            )

        # transformer encoder
        cfg_te = cfg.CONCAT_TRANSFORMER.transformer_encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg_lei.get("out_dim", 128),
                nhead=cfg_te.get("nhead", 10),
                dim_feedforward=cfg_te.get("dim_feedforward", 512),
                dropout=cfg_te.get("dropout", 0.1),
                activation=cfg_te.get("activation", "relu"),
            ),
            num_layers=cfg_te.get("num_layers", 2),
        )
        # linear encoder
        cfg_le = cfg.CONCAT_TRANSFORMER.linear_encoder_output
        self.linear_encoder = nn.Sequential()
        current_dim = self.seq_len * cfg_lei.get("out_dim", 128)
        in_dims = [current_dim] + cfg_le.get("hidden_encoder_layer_widths", [1024, 512])
        out_dims = in_dims[1:] + [2 * self.latent_dim]
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            self.linear_encoder.add_module(f"linear_{i}", nn.Linear(in_dim, out_dim))
            if i < len(in_dims) - 1:
                self.linear_encoder.add_module(
                    f"activation_{i}", activation_dict[cfg_le.get("activation", "relu")]
                )
                if self.batch_norm:
                    self.linear_encoder.add_module(
                        f"batch_norm_{i}", nn.BatchNorm1d(out_dim)
                    )
                self.linear_encoder.add_module(
                    f"dropout_{i}", nn.Dropout(cfg_le.get("dropout", 0.1))
                )

        # linear decoder
        cfg_ldi = cfg.CONCAT_TRANSFORMER.linear_decoder_input
        self.linear_decoder_input = nn.Sequential()
        target_dim = current_dim
        current_dim = self.latent_dim
        in_dims = [current_dim] + cfg_ldi.get(
            "hidden_decoder_layer_widths",
            [current_dim * 2, current_dim * 4, current_dim * 8],
        )
        out_dims = in_dims[1:] + [target_dim]
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            self.linear_decoder_input.add_module(
                f"linear_{i}", nn.Linear(in_dim, out_dim)
            )
            # if i>0:
            self.linear_decoder_input.add_module(
                f"activation_{i}", activation_dict[cfg_ldi.get("activation", "relu")]
            )
            if self.batch_norm:
                self.linear_decoder_input.add_module(
                    f"batch_norm_{i}", nn.BatchNorm1d(out_dim)
                )
            self.linear_decoder_input.add_module(
                f"dropout_{i}", nn.Dropout(cfg_ldi.get("dropout", 0.1))
            )

        # transformer decoder
        cfg_td = cfg.CONCAT_TRANSFORMER.transformer_decoder
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=cfg_lei.get("out_dim", 128),
                nhead=cfg_td.get("nhead", 10),
                dim_feedforward=cfg_td.get("dim_feedforward", 512),
                dropout=cfg_td.get("dropout", 0.1),
                activation=cfg_td.get("activation", "relu"),
            ),
            num_layers=cfg_td.get("num_layers", 3),
        )

        # linear output
        cfg_ldo = cfg.CONCAT_TRANSFORMER.linear_decoder_output
        self.linear_decoder_output = nn.Sequential()
        target_dim = (self.seq_len - 1) * 66 + 3 * self.seq_len + 66
        current_dim = cfg_lei.get("out_dim", 128) * self.seq_len
        in_dims = [current_dim] + cfg_ldo.get(
            "hidden_decoder_layer_widths",
            [target_dim // 8, target_dim // 4, target_dim // 2],
        )
        out_dims = in_dims[1:] + [target_dim]

        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            self.linear_decoder_output.add_module(
                f"linear_{i}", nn.Linear(in_dim, out_dim)
            )
            # if i>0:
            if i < len(in_dims) - 1:
                self.linear_decoder_output.add_module(
                    f"activation_{i}",
                    activation_dict[cfg_ldo.get("activation", "relu")],
                )

                if self.batch_norm:
                    self.linear_decoder_output.add_module(
                        f"batch_norm_{i}", nn.BatchNorm1d(out_dim)
                    )
                self.linear_decoder_output.add_module(
                    f"dropout_{i}", nn.Dropout(cfg_ldo.get("dropout", 0.1))
                )

        if self.load:
            print(f"Loading model from {self.checkpoint_path}")
            weights = torch.load(self.checkpoint_path)
            self.load_state_dict(weights["state_dict"])
            print("loaded model from:", self.checkpoint_path)

    def encode(self, pose0, velocity_relative, root_travel, verbose=False):
        # pose0: (batch_size, 1, num_joints=22, 3)
        # velocity_relative: (batch_size, seq_len-1, num_joints=22, 3)
        # root_travel: (batch_size, seq_len, 1, 3)

        # 1. encoder
        # concat
        # linear layer
        # transformer encoder

        x = torch.cat(
            [
                nn.Flatten()(velocity_relative),
                nn.Flatten()(root_travel),
                nn.Flatten()(pose0),
            ],
            dim=1,
        )
        if verbose:
            print(x.shape)
        x = self.linear_encoder_input(x)
        x = x.view(
            -1,
            self.seq_len,
            self.cfg.CONCAT_TRANSFORMER.linear_encoder_input.get("out_dim", 66),
        )
        if verbose:
            print(x.shape)
        # print(x.shape)
        x = self.transformer_encoder(x)
        # print(x.shape)
        x = x.view(
            -1,
            self.seq_len
            * self.cfg.CONCAT_TRANSFORMER.linear_encoder_input.get("out_dim", 128),
        )
        if verbose:
            print(x.shape)
        # print(x.shape)
        x = self.linear_encoder(x)
        if verbose:
            print(x.shape)
        mu, logvar = torch.chunk(x, 2, dim=1)
        # print(mu.shape, logvar.shape)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu.__add__(eps.__mul__(std))

    def decode(self, z, verbose=False):
        out_dim = self.cfg.CONCAT_TRANSFORMER.linear_encoder_input.get("out_dim", 128)
        if verbose:
            print("out_dim:", out_dim)
        # print('z shape:     ', z.shape)
        if verbose:
            print("z shape:     ", z.shape)
        z = self.linear_decoder_input(z)
        if verbose:
            print("z shape:     ", z.shape)
        # print('z shape:     ', z.shape)
        z = z.view(-1, self.seq_len, out_dim)
        if verbose:
            print("z shape:     ", z.shape)
        # print('z shape:     ', z.shape)

        z = self.transformer_decoder(z, z)
        if verbose:
            print("z shape:     ", z.shape)
        # print('z shape:     ', z.shape)
        z = z.view(-1, self.seq_len * out_dim)
        if verbose:
            print("z shape:     ", z.shape)
        # print('z shape:     ', z.shape)
        z = self.linear_decoder_output(z)
        if verbose:
            print("z shape:     ", z.shape)
        # vel
        # print(z.shape)
        pose0 = z[:, -66:].view(-1, 1, 22, 3)
        vel = z[:, : (self.seq_len - 1) * 66].view(-1, self.seq_len - 1, 22, 3)
        root = z[:, (self.seq_len - 1) * 66 : -66].view(-1, self.seq_len, 1, 3)

        vel = vel.view(-1, self.seq_len - 1, 22, 3)
        root = root.view(-1, self.seq_len, 1, 3)

        return pose0, vel, root

    def forward(self, pose0, velocity_relative, root_travel):
        mu, logvar = self.encode(pose0, velocity_relative, root_travel)
        z = self.reparameterize(mu, logvar)
        pose0_rec, vel_rec, root_rec = self.decode(z)
        return pose0_rec, vel_rec, root_rec, mu, logvar

    def reconstruct(self, pose0, velocity_relative, root_travel=None, verbose=False):
        if verbose:
            print("pose0:", pose0.shape)
        if verbose:
            print("velocity_relative:", velocity_relative.shape)
        motion_less_root = torch.cumsum(
            torch.cat([pose0, velocity_relative], dim=0), dim=0
        )
        if root_travel is not None:
            if verbose:
                print("root_travel:", root_travel.shape)
            return motion_less_root + root_travel
        return motion_less_root

    def reconstruction_step(self, res):
        idx = torch.randint(0, res["pose0"]["true"].shape[0], (1,)).item()
        # print('idx:', idx)
        recon = self.reconstruct(
            res["pose0"]["rec"][idx], res["vel"]["rec"][idx], res["root"]["rec"][idx]
        )
        recon = recon.cpu().detach().numpy()

        true = self.reconstruct(
            res["pose0"]["true"][idx], res["vel"]["true"][idx], res["root"]["true"][idx]
        )
        true = true.cpu().detach().numpy()
        return recon, true

    def training_step(self, batch, batch_idx):
        res = self._common_step(batch, batch_idx)

        loss = {k + "_trn": v for k, v in res["loss"].items()}
        self.log_dict(loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        # clip gradients --> do i do this here? # TODO
        # if self.clip > 0:
        #     torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
        return res["loss"]["total"]

    def validation_step(self, batch, batch_idx):
        res = self._common_step(batch, batch_idx)
        loss = {k + "_val": v for k, v in res["loss"].items()}
        self.log_dict(loss)

        if batch_idx == 0:
            print()
            recon, true = self.reconstruction_step(res)
            im_arr = plot_3d_motion_frames_multiple(
                [recon, true],
                ["recon", "true"],
                nframes=5,
                radius=2,
                figsize=(20, 8),
                return_array=True,
                velocity=False,
            )

            self.logger.experiment.add_image(
                "recon_vs_true", im_arr, global_step=self.global_step
            )
            if self.save_animations:
                # print("Saving animations")
                folder = self.logger.log_dir
                current_epoch = self.current_epoch

                plot_3d_motion_animation(
                    recon,
                    "recon",
                    figsize=(10, 10),
                    fps=20,
                    radius=2,
                    save_path=f"{folder}/recon_{current_epoch}.mp4",
                    velocity=False,
                    save_path_2=f"{folder}/recon_latest.mp4",
                )
                plt.close()

    def test_step(self, batch, batch_idx):
        res = self._common_step(batch, batch_idx)
        loss = {k + "_tst": v for k, v in res["loss"].items()}
        # self.log("test_loss", loss)
        # we want to add test loss final to the tensorboard
        self.log_dict(loss)
        idx = 0
        if batch_idx == idx and self.save_animations:
            recon, true = self.reconstruction_step(res)
            print("Saving animations")
            folder = self.logger.log_dir
            plot_3d_motion_animation(
                recon,
                "recon",
                figsize=(10, 10),
                fps=20,
                radius=2,
                save_path=f"{folder}/recon_test.mp4",
                velocity=False,
            )
            plt.close()
        return loss

    def _common_step(self, batch, batch_idx):
        pose0, vel, root = batch

        pose0_rec, vel_rec, root_rec, mu, logvar = self(pose0, vel, root)
        loss_data = {
            "velocity": {"true": vel, "rec": vel_rec, "weight": 10000},
            "root": {"true": root, "rec": root_rec, "weight": 0.01},
            "pose0": {"true": pose0, "rec": pose0_rec, "weight": 0.1},
        }
        loss = self.loss_function(loss_data, mu, logvar)
        return dict(
            loss=loss,
            pose0={"true": pose0, "rec": pose0_rec},
            vel={"true": vel, "rec": vel_rec},
            root={"true": root, "rec": root_rec},
            mu=mu,
            logvar=logvar,
        )

    def configure_optimizers(self):
        # this is also where we would put the scheduler
        return get_optimizer(self, self.optimizer, self.lr)
