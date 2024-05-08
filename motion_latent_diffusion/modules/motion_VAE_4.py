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

def lengths_to_mask(lengths: List[int],
                    device: torch.device,
                    max_len: int = None) -> Tensor:
    """
    Provides a mask, of length max_len or the longest element in lengths. With True for the elements less than the length for each length in lengths.
    """
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)

from typing import List, Optional
import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_clone(module):
    return copy.deepcopy(module)


class SkipTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, d_model=256):
        super().__init__()
        self.d_model = d_model

        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1

        num_block = (num_layers-1)//2
        self.input_blocks = _get_clones(encoder_layer, num_block)
        self.middle_block = _get_clone(encoder_layer)
        self.output_blocks = _get_clones(encoder_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2*self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        x = src

        xs = []
        for module in self.input_blocks:
            x = module(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            xs.append(x)

        x = self.middle_block(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            x = self.norm(x)
        return x
    


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation_dict[activation]
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, activation='relu', seq_len=120):
        super(TransformerEncoder, self).__init__()
        self.latent_size = 1   # 1 for single timestep
        self.latent_dim = d_model # 256
        self.seq_len = seq_len
        self.bite_size = 10
        # self.query_pos_encoder = nn.Embedding(1000, d_model)

        self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size * 2, self.latent_dim))

        self.skel_enc = nn.Linear(66, d_model)
        self.query_pos_encoder = PositionalEncoding(d_model, dropout, batch_first=True)
        self.query_pos_decoder = PositionalEncoding(d_model, dropout, batch_first=True)

        encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            False,
        )

        encoder_layer3 = TransformerEncoderLayer(
            self.latent_dim//16,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            False,
        )

        encoder_layer4 = TransformerEncoderLayer(
            self.latent_dim//16,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            False,
        )

        self.encoder = SkipTransformerEncoder(
            encoder_layer,
            norm=nn.LayerNorm(d_model),
            num_layers=num_layers,
            d_model=d_model
        )


        # nn.Linear(d_model*self.bite_size, d_model, device='mps')
        self.temporal_compressor = nn.Sequential(
            nn.Linear(d_model*self.bite_size, d_model*4),
            nn.LeakyReLU(),
            nn.Linear(d_model*4, d_model),
        )
        
        self.encoder_transformer2 = SkipTransformerEncoder(
            encoder_layer,
            norm=nn.LayerNorm(d_model),
            num_layers=num_layers,
            d_model=d_model
        )

        # nn.Linear(d_model, d_model//16)
        self.spatial_compressor = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.LeakyReLU(),
            nn.Linear(d_model*4, d_model//16)
        )

        self.encoder_transformer3 = SkipTransformerEncoder(
            encoder_layer3,
            norm=nn.LayerNorm(d_model//16),
            num_layers=num_layers,
            d_model=d_model//16
        )
        #  nn.Linear(42, 32)
        self.temporal_compressor2 =nn.Sequential(
            nn.Linear(42, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32)
        )

        self.final_encode_linear = nn.Linear(d_model*2, d_model*2)
        
        ### DECODER
        # nn.Linear(16, 42)
        self.temporal_decompressor1 = nn.Sequential(
            nn.Linear(16, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 42)

        )
        self.decoder_transformer1 = SkipTransformerEncoder( 
            encoder_layer4,
            norm=nn.LayerNorm(d_model//16),
            num_layers=num_layers,
            d_model=d_model//16
        )
        # nn.Linear(16, d_model)
        self.spatial_decompressor = nn.Sequential(
            nn.Linear(16, 128),
            nn.LeakyReLU(),
            nn.Linear(128, d_model)
        )

        self.decoder_transformer2 =  SkipTransformerEncoder( 
            encoder_layer,
            norm=nn.LayerNorm(d_model),
            num_layers=num_layers,
            d_model=d_model
        )
        # nn.Linear(42,seq_len)
        self.temporal_decompressor2 = nn.Sequential(
            nn.Linear(42, 128),
            nn.LeakyReLU(),
            nn.Linear(128, seq_len)
        )

        self.decoder = SkipTransformerEncoder(
            encoder_layer,
            norm=nn.LayerNorm(d_model),
            num_layers=num_layers,
            d_model=d_model
        )
        
        self.final_layer = nn.Linear(d_model, 66)
        
    def forward(self, src: Tensor):
        z, lengths, mu, logvar = self.encode(src)
        # mu, logvar = dist[:1], dist[1:]
        # z = self.reparameterize(mu, logvar)
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
        xseq = self.query_pos_encoder(xseq)
        dist = self.encoder(xseq, src_key_padding_mask=~aug_mask)#[:dist.shape[0]]
        # print(dist.shape)
        # temporal compression  8 --> 1
        # make small
        bite_size = self.bite_size
        # print(dist.shape)
        dists_bite_sized = [dist[i*bite_size:(i+1)*bite_size].permute(1, 0, 2).reshape(batch_size, -1)
                     for i in range(self.seq_len//bite_size)]
        
        dists_compressed = [self.temporal_compressor(d).view(batch_size, 1, -1) for d in dists_bite_sized]
        dists_compressed = torch.cat(dists_compressed, axis=1).permute(1, 0, 2)
        # print(dists_compressed.shape)
        dists = self.encoder_transformer2(dists_compressed)
        # print(dists.shape)

        # spatial compression
        # print('beginning spatial compression')
        dists = dists.permute(1, 0, 2)
        dists = self.spatial_compressor(dists)
        dists = dists.permute(1, 0, 2)
        # print(dists.shape)

        # transformer 3
        # print('beginning transformer 3')
        # print('shape before:', dists.shape)
        dists = self.encoder_transformer3(dists)
        dists = dists.permute(1, 0, 2)
        # print('shape after:', dists.shape)

        # more temporal compression
        # print('beginning more temporal compression')
        dists = dists.permute(0,2,1)
        dists = self.temporal_compressor2(dists)
        # print(dists.shape)

        dist = torch.flatten(dists, start_dim=1)

        

        dist = self.final_encode_linear(dist)

        # print(dist.shape)

        mu, logvar = dist[:, :self.latent_dim], dist[:, self.latent_dim:]


        # resample
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample().unsqueeze(0)

        return latent, lengths, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: Tensor, lengths: List[int]):
        # print('decoding')
        mask = lengths_to_mask(lengths, z.device)
        bs, nframes = mask.shape
        # print('z.shape', z.shape)
        z = z.view(16, bs, 16).permute(1,2,0)
        z = self.temporal_decompressor1(z)
        # print('z.shape', z.shape)
        z = z.permute(2,0,1)
        z = self.decoder_transformer1(z)
        # print(z.shape)

        # spatial decompressor
        z = self.spatial_decompressor(z)

        # transfomer
        z = self.decoder_transformer2(z)
        # print(z.shape)
        
        # temporal_decompressor 2
        z = z.permute(1,2,0)
        z = self.temporal_decompressor2(z)
        z = z.permute(2, 0, 1)
        # print(z.shape)



        # queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)
        # xseq = torch.cat((z, queries), axis=0)
        xseq = z
        z_mask = torch.ones((bs, self.latent_size),
                            dtype=bool,
                            device=z.device)
        augmask = torch.cat((z_mask, mask), axis=1)


        xseq = self.query_pos_decoder(xseq)
        output = self.decoder(xseq)#[z.shape[0]:]
        
        # print(output.shape)
        output = self.final_layer(output)
        # output[~mask.T] = 0
        feats = output.permute(1, 0, 2)
        # print('feats.shape', feats.shape)
        # zero for padded area
        # print('mask.shape', mask.shape)
        # feats[~mask] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        
        return feats
    

class TransformerMotionVAE(pl.LightningModule):
    def __init__(self, **kwargs):
        super(TransformerMotionVAE, self).__init__()
        self.lr = kwargs['learning_rate']
        self.save_animations_freq = kwargs['_save_animations_freq']
        self.seq_len = kwargs['seq_len']

        self.model = TransformerEncoder(
            d_model=kwargs['latent_dim'],
            nhead=kwargs['nhead'],
            num_layers=kwargs['num_layers'],
            dim_feedforward=kwargs['dim_feedforward'],
            dropout=kwargs['dropout'],
            activation=kwargs['transformer_activation'],
            seq_len=self.seq_len
        )  # TODO: dont send all kwargs

        self.criterion = VAE_Loss(kwargs['loss_weights'])
        self.val_outputs = {}

    def forward(self, x):
        out, mu, logvar = self.model(x)
        return out, mu, logvar
    
    def motion_seq_decomposition(self, motion_seq_batched):
        bs = motion_seq_batched.shape[0]
        seq = motion_seq_batched
        velocity = torch.diff(seq, dim=1)
        if seq.dim() == 3:
            seq = seq.view(seq.shape[0], seq.shape[1], 22, 3)
        root = seq[:, :, :1]
        # print('root.shape', root.shape)
        # print('seq.shape', seq.shape)
        motion_relative = seq - root
        pose_0 = motion_relative[:, 0].view(bs, 1, -1)
        # print('pose_0.shape', pose_0.shape)
        # print('motion_relative.shape', motion_relative.shape)
        # print('velocity.shape', velocity.shape)
        
        return root, motion_relative, velocity, pose_0
    
    def _common_step(self, batch, batch_idx):
        
        pred_velocity = False

        motion_seq, text  = batch
        bs = motion_seq.shape[0]
        
        motion_seq = motion_seq.view(motion_seq.shape[0], motion_seq.shape[1], -1)

        root_gt, motion_relative_gt, vel_gt, pose_0 = self.motion_seq_decomposition(motion_seq)


        if pred_velocity:
            in_ = torch.cat((pose_0, vel_gt), 1)
        else:
            in_ = motion_seq

        out, mu, logvar = self(in_)


        # if predicting velocity
        if pred_velocity:
            pose_0_pred = out[:, 0].view(bs, 1, -1)
            vel_pred = out[:, 1:]  # [B, S-1, 66]
            # print('vel_pred.shape', vel_pred.shape)
            motion_seq_pred = torch.cumsum(vel_pred, dim=1)
            # print('motion_seq_pred.shape', motion_seq_pred.shape)
            # print('motion_seq.shape', motion_seq.shape)
            root_pred, motion_relative_pred, _, _ = self.motion_seq_decomposition(motion_seq_pred)
            
        else:
            root_pred, motion_relative_pred, vel_pred, pose_0_pred = self.motion_seq_decomposition(out)
            motion_seq_pred = out
        


        loss_dict = {
                "MOTION_L2": {"rec": out, "true": motion_seq},
                "DIVERGENCE_KL": {"mu": mu, "logvar": logvar},
                "ROOT_L2": {"rec": root_pred, "true": root_gt},
                "MOTIONRELATIVE_L2": {"rec": motion_relative_pred, "true": motion_relative_gt},
                "VELOCITY_L2": {"rec": vel_pred, "true": vel_gt},
                "POSE0_L2": {"rec": pose_0_pred, "true": pose_0},
            }
        # for k, v in loss_dict.items():
        #     for kk, vv in v.items():
        #         print(k, kk, vv.shape)

        loss, losses_scaled, losses_unscaled = self.criterion(loss_dict)

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
    

    def log_images_arr(self, recon_seq, motion_seq, nframes=5):
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
        
    def save_animations_func(self, recon_seq, motion_seq):
        self.folder = self.logger.log_dir
        self.subfolder = f"{self.folder}/animations"
        if not os.path.exists(self.subfolder):  # check if subfolder exists
            os.makedirs(self.subfolder)

        fname = f"{self.subfolder}/recon_epoch{self.current_epoch}.mp4"
        plot_3d_motion_animation(
            recon_seq, "recon", figsize=(10, 10), fps=20, radius=2, save_path=fname, velocity=False,)
        plt.close()

        
        shutil.copyfile(fname, f"{self.folder}/recon_latest.mp4")  # copy file to latest

        if self.current_epoch == 0:
            plot_3d_motion_animation(
                motion_seq, "true", figsize=(10, 10), fps=20, radius=2, save_path=f"{self.folder}/recon_true.mp4", velocity=False)
            plt.close()

    def on_validation_epoch_end(self):
        print('validation_epoch_end')
        motion_seq, recon_seq = self.val_outputs['motion_seq'], self.val_outputs['recon_seq']
        motion_seq = motion_seq.cpu().detach().numpy()
        recon_seq = recon_seq.cpu().detach().numpy()

        self.log_images_arr(recon_seq, motion_seq)

        self.folder = self.logger.log_dir
        self.subfolder = f"{self.folder}/animations"
        if not os.path.exists(self.subfolder):  # check if subfolder exists
            os.makedirs(self.subfolder)
        
        if (self.save_animations_freq != -1) and (self.current_epoch % self.save_animations_freq == 0):
            self.save_animations_func(recon_seq, motion_seq)
            
    def test_step(self, batch, batch_idx):
        res= self._common_step(batch, batch_idx)
        self.log("test_loss", res['loss'], prog_bar=True)
        self.log_dict(res['losses_unscaled'], prog_bar=True)

        return res['loss']
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
