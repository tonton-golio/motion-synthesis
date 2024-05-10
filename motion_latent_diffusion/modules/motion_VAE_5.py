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
from typing import List, Optional
import copy

def lengths_to_mask(lengths: List[int],
                    device: torch.device,
                    max_len: int = None) -> Tensor:
    """
    Provides a mask, of length max_len or the longest element in lengths. With True for the elements less than the length for each length in lengths.
    """
    # lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else lengths.max()
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

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
                           src_key_padding_mask=src_key_padding_mask)
            xs.append(x)

        x = self.middle_block(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, 
                           )

        if self.norm is not None:
            x = self.norm(x)
        return x

class CascadingTransformerAutoEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, activation='relu', seq_len=120, verbose=False):
        super(CascadingTransformerAutoEncoder, self).__init__()
        self.latent_size = 1   # 1 for single timestep
        self.latent_dim = d_model # 256
        self.seq_len = seq_len
        self.verbose = verbose
        self.conv1_out_channels = 32
        self.activation = nn.LeakyReLU()
        
        # ENCODER
        self.skel_enc = nn.Linear(66, d_model)
    
        self.skip_trans_enc = SkipTransformerEncoder(
            encoder_layer= nn.TransformerEncoderLayer(
                d_model=256, nhead=64, dim_feedforward=1024, 
                dropout=0.1, activation='gelu', 
                norm_first=False, batch_first=True),
            num_layers=7,
            norm=nn.LayerNorm(256),
            d_model=256
        )

        self.conv2d_enc = nn.Conv2d(
                            in_channels=1,
                            out_channels=self.conv1_out_channels,
                            kernel_size=(8, 256),
                            stride=(6, 1),
                            padding=(0, 0))
        
        self.enc_final_linear = nn.Sequential(
            nn.Linear(2208, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
        )
        # DECODER
        self.linear_dec = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 2208),
            nn.LeakyReLU(),
        )

        self.transconv2d_dec = nn.ConvTranspose2d(
                            in_channels=1,
                            out_channels=7,
                            kernel_size=8,
                            stride=(6,1),
                            padding=(0,0), 
                            output_padding=(4,0))

        self.linear_dec2 = nn.Sequential(
            nn.Linear(273, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
        )
        #nn.Linear(220, 256)

        self.skip_trans_dec2 = SkipTransformerEncoder(
            encoder_layer= nn.TransformerEncoderLayer(
                d_model=256, nhead=64, dim_feedforward=1024,
                dropout=0.1, activation='gelu',
                norm_first=False, batch_first=True),
            num_layers=7,
            norm=nn.LayerNorm(256),
            d_model=256
        )
        
        self.final_layer = nn.Linear(256, 66)
        
    def forward(self, src: Tensor):
        z, lengths, mu, logvar = self.encode(src)
        # mu, logvar = dist[:1], dist[1:]
        # z = self.reparameterize(mu, logvar)
        output = self.decode(z, lengths)
        return output, mu, logvar, lengths  

    def encode(self, src):
        # get lengths
        lengths = torch.tensor([len(feature) for feature in src], dtype=torch.float32).to(src.device)


        if self.verbose: print('ENCODING')
        # get shapes
        bs, nframes, nfeats = src.shape
        if self.verbose: print('batch size:', bs, 'nframes:', nframes, 'nfeats:', nfeats)

        # skeletal embedding
        x = self.skel_enc(src)
        if self.verbose: print('skel_enc:', x.shape)

        # pass through transformerencoder with skip connections
        x = self.skip_trans_enc(x)
        if self.verbose: print('skip trans enc:', x.shape)

        # make small with conv2d
        x = x.unsqueeze(1)
        x = self.conv2d_enc(x)
        x = self.activation(x)
        if self.verbose: print('conv2d:', x.shape)

        # map linear
        x = torch.flatten(x, start_dim=1)
        if self.verbose: print('flattened:', x.shape)

        # map linear
        x = self.enc_final_linear(x)
        if self.verbose: print('final linear:', x.shape)

        mu, logvar = x[:, :256], x[:, 256:]
        # resample
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()

        if self.verbose: print('latent:', latent.shape)
        latentdim = torch.prod(torch.tensor(latent.shape[1:]))
        if self.verbose: print('latentdim:', latentdim)
        return latent, lengths, mu, logvar
    
    
    def decode(self, z: Tensor, lengths: List[int]):
        if self.verbose: print('DECODING')
        mask = lengths_to_mask(lengths, z.device, self.seq_len)
        bs, nframes = mask.shape
        if self.verbose: print('batch size:', bs, 'nframes:', nframes, 'z shape:', z.shape)

        # map linear
        z = self.linear_dec(z)
        if self.verbose: print('linear:', z.shape)
        z = z.view(bs, 1, 69, 32)

        if self.verbose: print('linear:', z.shape)
        # expand with convtranspose2d
        z = self.transconv2d_dec(z)
        z = self.activation(z)
        if self.verbose: print('transconv1d:', z.shape)

  
        # map linear
        z = z.permute(0, 2, 1, 3).flatten(start_dim=2)
        z = self.linear_dec2(z)
        z = self.activation(z)
        if self.verbose: print('linear:', z.shape)

        # apply transformer
        z = self.skip_trans_dec2(z)
        if self.verbose: print('skip trans dec2:', z.shape)
        
        # final layer
        output = self.final_layer(z)
        if self.verbose: print('final layer:', output.shape)
        output[~mask] = 0
        feats = output#.permute(1, 0, 2)
        if self.verbose: print('feats:', feats.shape)

        return feats
class TransformerMotionVAE(pl.LightningModule):
    def __init__(self, **kwargs):
        super(TransformerMotionVAE, self).__init__()
        self.lr = kwargs['learning_rate']
        self.save_animations_freq = kwargs['_save_animations_freq']
        self.seq_len = kwargs['seq_len']

        self.model = CascadingTransformerAutoEncoder(
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

        out, mu, logvar, lengths = self.model(x)
        return out, mu, logvar, lengths
    
    def motion_seq_decomposition(self, motion_seq_batched):
        bs = motion_seq_batched.shape[0]
        seq = motion_seq_batched
        # print('seq.shape', seq.shape)
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

        out, mu, logvar, lengths = self(in_)


        # if predicting velocity
        if pred_velocity:
            pose_0_pred = out[:, 0].view(bs, 1, -1)
            motion_seq_pred = torch.cumsum(out, dim=1)
            vel_pred = out[:, 1:]  # [B, S-1, 66]
            # print('vel_pred.shape', vel_pred.shape)
            
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

        loss, losses_scaled, losses_unscaled = self.criterion(loss_dict, lengths=lengths)

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
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        # return [optimizer], [scheduler]
        return optimizer
