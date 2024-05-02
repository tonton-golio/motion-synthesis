import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import (
    plot_3d_motion_frames_multiple,
    plot_3d_motion_animation,
    plot_3d_motion_frames_multiple,
    activation_dict,
)
from glob import glob
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
from modules.loss import VAE_Loss

from torch import Tensor
from typing import List

# no activation class
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        latent_dim,
        seq_len,
        input_dim,
        n_layers,
        n_heads,
        dropout,
        hidden_dim_trans,
        transformer_activation,
        activation,
    ):
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
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 8),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 8, self.input_dim * self.seq_len),
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

def lengths_to_mask(lengths: List[int],
                    device: torch.device,
                    max_len: int = None) -> Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerEncoder, self).__init__()
        self.latent_size = 1   # 1 for single timestep
        self.latent_dim = d_model # 256
        
        # self.query_pos_encoder = nn.Embedding(1000, d_model)

        self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size * 2, self.latent_dim))


        self.skel_enc = nn.Linear(66, d_model)

        encoder_layer =  nn.TransformerEncoderLayer(  # TODO: Should we use full transformer instead?
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout, 
                activation=activation, 
                batch_first=False
                )
        encoder_norm = nn.LayerNorm(d_model)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            norm=encoder_norm,
            num_layers=num_layers)
        

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            activation=activation, 
            batch_first=False
            )
        decoder_norm = nn.LayerNorm(d_model)

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            norm=decoder_norm,
            num_layers=num_layers)
        
        self.final_layer = nn.Linear(d_model, 66)
        
    def forward(self, src: Tensor):
        dist, lengths = self.encode(src)
        mu, logvar = dist[:1], dist[1:]
        z = self.reparameterize(mu, logvar)
        output = self.decode(z, lengths)
        return output, mu, logvar

    def encode(self, src):
        batch_size = src.shape[0]
        lengths = [len(feature) for feature in src]
        x = src.permute(1, 0, 2)  # (B, S, F) -> (S, B, F)
        device = x.device
        mask = lengths_to_mask(lengths, device)
        dist = torch.tile(self.global_motion_token[:, None, :], (1, batch_size, 1))

        dist_masks = torch.ones((batch_size, dist.shape[0]),
                                dtype=bool,
                                device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)

        # adding the embedding token for all sequences
        x = self.skel_enc(x)
        xseq = torch.cat((dist, x), 0)
        # print(xseq.shape)
        # # xseq = self.query_pos_encoder(xseq)
        # print(xseq.shape)

        # xseq = self.query_pos_encoder(xseq)
        dist = self.encoder(xseq,
                            src_key_padding_mask=~aug_mask)[:dist.shape[0]]

        return dist, lengths
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: Tensor, lengths: List[int]):
        # print('decoding')
        mask = lengths_to_mask(lengths, z.device)
        bs, nframes = mask.shape

        queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)


        # print(z.shape)
        # print(queries.shape)
        xseq = torch.cat((z, queries), axis=0)
        z_mask = torch.ones((bs, self.latent_size),
                            dtype=bool,
                            device=z.device)
        augmask = torch.cat((z_mask, mask), axis=1)
        # xseq = self.query_pos_decoder(xseq)
        output = self.decoder( xseq, xseq)[z.shape[0]:]
        # print(output.shape)
        output = self.final_layer(output)
        # zero for padded area
        output[~mask.T] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        return feats
    

class TransformerMotionVAE(pl.LightningModule):
    def __init__(self, **kwargs):
        super(TransformerMotionVAE, self).__init__()
        self.lr = kwargs['learning_rate']
        self.save_animations = kwargs['_save_animations']

        self.model = TransformerEncoder(
            d_model=kwargs['latent_dim'],
            nhead=kwargs['nhead'],
            num_layers=kwargs['num_layers'],
            dim_feedforward=kwargs['dim_feedforward'],
            dropout=kwargs['dropout'],
            activation=kwargs['transformer_activation']
        )  # TODO: dont send all kwargs

        self.criterion = VAE_Loss(kwargs['loss_weights'])

        self.save_hyperparameters()


        self.val_outputs = {}

    def forward(self, x):
        out, mu, logvar = self.model(x)
        return out, mu, logvar
    
    def _common_step(self, batch, batch_idx):
        motion_seq, text  = batch
        motion_seq = motion_seq.view(motion_seq.shape[0], motion_seq.shape[1], -1)
        out, mu, logvar = self(motion_seq)
        loss, losses_scaled, losses_unscaled = self.criterion(
            {
                "MOTION_L2": {"rec": out, "true": motion_seq},
                "DIVERGENCE_KL": {"mu": mu, "logvar": logvar},
            }
        )
        return dict(
            loss=loss,
            losses_scaled=losses_scaled,
            losses_unscaled=losses_unscaled,
            motion_seq=motion_seq,
            recon_seq=out,
        )
    
    def training_step(self, batch, batch_idx):
        res = self._common_step(batch, batch_idx)
        self.log("train_loss", res['loss'], prog_bar=True)
        self.log_dict(res['losses_unscaled'], prog_bar=True)
        return res['loss']
    
    def validation_step(self, batch, batch_idx):
        res = self._common_step(batch, batch_idx)
        self.log("train_loss", res['loss'], prog_bar=True)
        self.log_dict(res['losses_unscaled'], prog_bar=True)

        self.val_outputs['motion_seq'] = res['motion_seq'][0]
        self.val_outputs['recon_seq'] = res['recon_seq'][0]
    
    def on_validation_epoch_end(self):
        print('validation_epoch_end')
        motion_seq, recon_seq = self.val_outputs['motion_seq'], self.val_outputs['recon_seq']
        motion_seq = motion_seq.cpu().detach().numpy()
        recon_seq = recon_seq.cpu().detach().numpy()
        self.folder = self.logger.log_dir
        self.subfolder = f"{self.folder}/animations"
        if not os.path.exists(self.subfolder):  # check if subfolder exists
            os.makedirs(self.subfolder)
        im_arr = plot_3d_motion_frames_multiple(
                [recon_seq, motion_seq],
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
            fname = f"{self.subfolder}/recon_epoch{self.current_epoch}.mp4"
            plot_3d_motion_animation(
                recon_seq, "recon", figsize=(10, 10), fps=20, radius=2, save_path=fname, velocity=False,)
            plt.close()

            
            shutil.copyfile(fname, f"{self.folder}/recon_latest.mp4")  # copy file to latest

            if self.current_epoch == 0:
                plot_3d_motion_animation(
                    motion_seq, "true", figsize=(10, 10), fps=20, radius=2, save_path=f"{self.folder}/recon_true.mp4", velocity=False)
                plt.close()

    def test_step(self, batch, batch_idx):
        res= self._common_step(batch, batch_idx)
        self.log("test_loss", res['loss'], prog_bar=True)
        self.log_dict(res['losses_unscaled'], prog_bar=True)

        return res['loss']
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
