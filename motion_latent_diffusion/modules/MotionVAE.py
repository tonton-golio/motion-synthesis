import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl

import matplotlib.pyplot as plt
import numpy as np
import os, shutil, copy
from torch import Tensor
from typing import List, Optional

from modules.Loss import VAE_Loss
from utils import (
    plot_3d_motion_frames_multiple,
    plot_3d_motion_animation,
    activation_dict,
)



#### VAE 1
class VAE1(nn.Module):
    def __init__(self, **kwargs):
        super(VAE1, self).__init__()
        self.verbose = False
        self.input_dim = kwargs.get("input_dim", 66)
        self.nhead = kwargs.get("nhead")
        self.ff_transformer = kwargs.get("ff_transformer")
        self.dropout = kwargs.get("dropout")
        self.transformer_activation = kwargs.get("transformer_activation")
        self.nlayers_transformer = kwargs.get("nlayers_transformer")
        
        self.seq_len = kwargs.get("seq_len")
        self.activation = activation_dict[kwargs.get("activation")]
        self.hidden_dim_linear = kwargs.get("hidden_dim_linear")
        self.latent_dim = kwargs.get("latent_dim")

        self.setup_model()
        self._reset_parameters()

    def setup_model(self):
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.input_dim,
                nhead=self.nhead,
                dim_feedforward=self.ff_transformer,
                dropout=self.dropout,
                batch_first=True,
                activation=self.transformer_activation,
                norm_first=True,
            ),
            num_layers=self.nlayers_transformer,
            norm=nn.LayerNorm(self.input_dim),
        )

        self.encoder_linear_block = nn.Sequential(
            nn.Linear(self.input_dim * self.seq_len, 8 * self.hidden_dim_linear),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim_linear * 8, self.hidden_dim_linear * 4),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim_linear * 4, self.hidden_dim_linear * 2),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim_linear * 2, 2 * self.latent_dim),
        )

        self.decoder_linear_block = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim_linear),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim_linear, self.hidden_dim_linear * 4),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim_linear * 4, self.hidden_dim_linear * 8),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim_linear * 8, self.input_dim * self.seq_len),
            self.activation,
        )

        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.input_dim,
                nhead=self.nhead,
                dim_feedforward=self.ff_transformer,
                dropout=self.dropout,
                batch_first=True,
                activation=self.transformer_activation,
                norm_first=True,
            ),
            num_layers=self.nlayers_transformer,
        )

        self.output_block = nn.Sequential(
            # reshape: x = x.view(-1, self.input_dim * self.seq_len)
            nn.Flatten(),
            nn.Linear(self.input_dim * self.seq_len, self.input_dim * self.seq_len),
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        z, _, mu, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, z, mu, logvar
    
    def encode(self, x):
        x = x.view(-1, self.seq_len, self.input_dim)
        x = self.transformer_encoder(x)
        x = x.view(-1, self.input_dim * self.seq_len)
        x = self.encoder_linear_block(x)
        mu, logvar = x.chunk(2, dim=1)
        z = self.reparametrize(mu, logvar)
        return z, None, mu, logvar

    def decode(self, z):
        x = self.decoder_linear_block(z)
        x = x.view(-1, self.seq_len, self.input_dim)
        x = self.transformer_decoder(x, x)
        x = self.output_block(x)
        x = x.view(-1, self.seq_len, self.input_dim//3, 3)
        return x

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

#### VAE 4
def lengths_to_mask(lengths: List[int],
                    device: torch.device,
                    max_len: int = None) -> Tensor:
    """
    Provides a mask, of length max_len or the longest element in lengths. With True for the elements less than the length for each length in lengths.
    """
    if type(lengths) == list:
        lengths = torch.tensor(lengths, device=device)
    else:
        lengths = lengths.to(device)
        
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

class VAE4(nn.Module):
    def __init__(self, **kwargs):
        super(VAE4, self).__init__()
        self.verbose = False

        self.latent_size = 1   # 1 for single timestep
        self.latent_dim = kwargs.get("latent_dim", 256)
        self.seq_len = kwargs.get("seq_len")

        self.dropout = kwargs.get("dropout", 0.1)
        self.nhead = kwargs.get("nhead")

        self.transformer_activation = kwargs.get("transformer_activation")
        self.nlayers_transformer = kwargs.get("nlayers_transformer")
        
        self.seq_len = kwargs.get("seq_len")
        self.activation = activation_dict[kwargs.get("activation")]
        
        self.ff_transformer = kwargs.get("ff_transformer")

        self.bite_size = 10
        # self.query_pos_encoder = nn.Embedding(1000, d_model)
        
        self.setup_model()
        self._reset_parameters()
    
    def setup_model(self):
        self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size * 2, self.latent_dim))

        self.skel_enc = nn.Linear(66, self.latent_dim)
        self.query_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, batch_first=True)
        self.query_pos_decoder = PositionalEncoding(self.latent_dim, self.dropout, batch_first=True)

        encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            self.nhead,
            self.ff_transformer,
            self.dropout,
            self.transformer_activation,
            False,
        )

        encoder_layer3 = TransformerEncoderLayer(
            self.latent_dim//16,
            self.nhead,
            self.ff_transformer,
            self.dropout,
            self.transformer_activation,
            False,
        )

        encoder_layer4 = TransformerEncoderLayer(
            self.latent_dim//16,
            self.nhead,
            self.ff_transformer,
            self.dropout,
            self.transformer_activation,
            False,
        )

        self.encoder = SkipTransformerEncoder(
            encoder_layer,
            norm=nn.LayerNorm(self.latent_dim),
            num_layers=self.nlayers_transformer,
            d_model=self.latent_dim
        )


        # nn.Linear(self.latent_dim*self.bite_size, self.latent_dim, device='mps')
        self.temporal_compressor = nn.Sequential(
            nn.Linear(self.latent_dim*self.bite_size, self.latent_dim*4),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim*4, self.latent_dim),
        )
        
        self.encoder_transformer2 = SkipTransformerEncoder(
            encoder_layer,
            norm=nn.LayerNorm(self.latent_dim),
            num_layers=self.nlayers_transformer,
            d_model=self.latent_dim
        )

        # nn.Linear(self.latent_dim, self.latent_dim//16)
        self.spatial_compressor = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim*4),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim*4, self.latent_dim//16)
        )

        self.encoder_transformer3 = SkipTransformerEncoder(
            encoder_layer3,
            norm=nn.LayerNorm(self.latent_dim//16),
            num_layers=self.nlayers_transformer,
            d_model=self.latent_dim//16
        )
        #  nn.Linear(42, 32)
        self.temporal_compressor2 =nn.Sequential(
            nn.Linear(42, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32)
        )

        self.final_encode_linear = nn.Linear(self.latent_dim*2, self.latent_dim*2)
        
        ### DECODER
        # nn.Linear(16, 42)
        self.temporal_decompressor1 = nn.Sequential(
            nn.Linear(16, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 42)

        )
        self.decoder_transformer1 = SkipTransformerEncoder( 
            encoder_layer4,
            norm=nn.LayerNorm(self.latent_dim//16),
            num_layers=self.nlayers_transformer,
            d_model=self.latent_dim//16
        )
        # nn.Linear(16, self.latent_dim)
        self.spatial_decompressor = nn.Sequential(
            nn.Linear(16, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.latent_dim)
        )

        self.decoder_transformer2 =  SkipTransformerEncoder( 
            encoder_layer,
            norm=nn.LayerNorm(self.latent_dim),
            num_layers=self.nlayers_transformer,
            d_model=self.latent_dim
        )
        # nn.Linear(42,seq_len)
        self.temporal_decompressor2 = nn.Sequential(
            nn.Linear(42, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.seq_len)
        )

        self.decoder = SkipTransformerEncoder(
            encoder_layer,
            norm=nn.LayerNorm(self.latent_dim),
            num_layers=self.nlayers_transformer,
            d_model=self.latent_dim
        )
        
        self.final_layer = nn.Linear(self.latent_dim, 66)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor):
        z, lengths, mu, logvar = self.encode(src)
        recon = self.decode(z, lengths)
        return recon, z, mu, logvar

    def encode(self, src):
        src = src.view(-1, self.seq_len, 66)
        batch_size = src.shape[0]
        lengths = [len(feature) for feature in src]
        x = src.permute(1, 0, 2)  # (B, S, F) -> (S, B, F)
        device = x.device
        mask = lengths_to_mask(lengths, device)
        dist = torch.tile(self.global_motion_token[:, None, :], (1, batch_size, 1))

        dist_masks = torch.ones((batch_size, dist.shape[0]), dtype=bool, device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)

        x = self.skel_enc(x)# adding the embedding token for all sequences

        xseq = torch.cat((dist, x), 0)
        xseq = self.query_pos_encoder(xseq)
        dist = self.encoder(xseq, src_key_padding_mask=~aug_mask)#[:dist.shape[0]]

        # temporal compression  8 --> 1
        bite_size = self.bite_size

        dists_bite_sized = [dist[i*bite_size:(i+1)*bite_size].permute(1, 0, 2).reshape(batch_size, -1)
                     for i in range(self.seq_len//bite_size)]
        
        dists_compressed = [self.temporal_compressor(d).view(batch_size, 1, -1) for d in dists_bite_sized]
        dists_compressed = torch.cat(dists_compressed, axis=1).permute(1, 0, 2)

        dists = self.encoder_transformer2(dists_compressed)


        # spatial compression

        dists = dists.permute(1, 0, 2)
        dists = self.spatial_compressor(dists)
        dists = dists.permute(1, 0, 2)

        # transformer 3

        dists = self.encoder_transformer3(dists)
        dists = dists.permute(1, 0, 2)

        # more temporal compression
        dists = dists.permute(0,2,1)
        dists = self.temporal_compressor2(dists)


        dist = torch.flatten(dists, start_dim=1)

        dist = self.final_encode_linear(dist)
        mu, logvar = dist[:, :self.latent_dim], dist[:, self.latent_dim:]

        # resample
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample().unsqueeze(0)

        return latent, lengths, mu, logvar

    def decode(self, z: Tensor, lengths: List[int]):
        mask = lengths_to_mask(lengths, z.device)
        bs, nframes = mask.shape

        z = z.view(16, bs, 16).permute(1,2,0)
        z = self.temporal_decompressor1(z)

        z = z.permute(2,0,1)
        z = self.decoder_transformer1(z)

        # spatial decompressor
        z = self.spatial_decompressor(z)

        # transfomer
        z = self.decoder_transformer2(z)

        
        # temporal_decompressor 2
        z = z.permute(1,2,0)
        z = self.temporal_decompressor2(z)
        z = z.permute(2, 0, 1)

        # queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)
        # xseq = torch.cat((z, queries), axis=0)
        xseq = z
        z_mask = torch.ones((bs, self.latent_size),
                            dtype=bool,
                            device=z.device)
        augmask = torch.cat((z_mask, mask), axis=1)


        xseq = self.query_pos_decoder(xseq)
        output = self.decoder(xseq)#[z.shape[0]:]

        output = self.final_layer(output)
        # output[~mask.T] = 0
        feats = output.permute(1, 0, 2)

        # zero for padded area

        # feats[~mask] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = feats.view(bs, nframes, 22, 3) 
        return feats
    

#### VAE 5
class SkipTransformerEncoder2(nn.Module):
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
    
class VAE5(nn.Module):
    def __init__(self, **kwargs):
        super(VAE5, self).__init__()
        self.verbose = False

        self.latent_size = 1   # 1 for single timestep
        self.latent_dim = kwargs.get("latent_dim", 256)
        self.seq_len = kwargs.get("seq_len")
        
        self.conv1_out_channels = 32
        self.activation = nn.LeakyReLU()
        
        self.setup_model()
        self._reset_parameters()

    def setup_model(self):
        # ENCODER
        self.skel_enc = nn.Linear(66, self.latent_dim)
    
        self.skip_trans_enc = SkipTransformerEncoder2(
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

        self.skip_trans_dec2 = SkipTransformerEncoder2(
            encoder_layer= nn.TransformerEncoderLayer(
                d_model=256, nhead=64, dim_feedforward=1024,
                dropout=0.1, activation='gelu',
                norm_first=False, batch_first=True),
            num_layers=7,
            norm=nn.LayerNorm(256),
            d_model=256
        )
        
        self.final_layer = nn.Linear(256, 66)
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor):
        z, lengths, mu, logvar = self.encode(src)
        recon = self.decode(z, lengths)
        return recon, z, mu, logvar

    def encode(self, src):

        src = src.view(-1, self.seq_len, 66)
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
        z = dist.rsample()

        if self.verbose: print('latent:', z.shape)
        latentdim = torch.prod(torch.tensor(z.shape[1:]))
        if self.verbose: print('latentdim:', latentdim)

        return z, lengths, mu, logvar
    
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

        feats = feats.view(bs, nframes, 22, 3)
        return feats

## VAE 6
class PositionEmbeddingSine1D(nn.Module):

    def __init__(self, d_model, max_len=500, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            pos = self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            pos = self.pe[:x.shape[0], :]
        return pos

class VAE6(nn.Module):
    def __init__(self, **kwargs):
        super(VAE6, self).__init__()
        self.verbose = False

        self.latent_size = 1   # 1 for single timestep
        self.latent_dim = kwargs.get("latent_dim", 256)
        self.seq_len = kwargs.get("seq_len")
        self.nhead = kwargs.get("nhead")
        self.dropout = kwargs.get("dropout")
        self.transformer_activation = kwargs.get("transformer_activation")
        self.nlayers_transformer = kwargs.get("nlayers_transformer")
        self.activation = nn.LeakyReLU()
        
        self.setup_model()
        self._reset_parameters()

    def setup_model(self):
        ld = self.latent_dim
        
        self.skel_enc = nn.Linear(66, ld)
    
        self.skip_trans_enc = SkipTransformerEncoder2(
            encoder_layer= nn.TransformerEncoderLayer(
                d_model=ld, nhead=self.nhead, dim_feedforward=4*ld, 
                dropout=self.dropout, activation=self.transformer_activation, 
                norm_first=False, batch_first=True),
            num_layers=self.nlayers_transformer,
            norm=nn.LayerNorm(ld),
            d_model=ld
        )

        self.latent_enc_linear = nn.Sequential(
            nn.Linear(2*ld, 2*ld),
            nn.LeakyReLU(),
            nn.Linear(2*ld, 2*ld),
        )   
        self.query_pos_decoder = PositionEmbeddingSine1D(ld, max_len=self.seq_len, batch_first=True)

        self.trans_dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=ld, nhead=self.nhead, dim_feedforward=4*ld,
                dropout=self.dropout, activation=self.transformer_activation,
                norm_first=False, batch_first=True),
            num_layers=self.nlayers_transformer,
            norm=nn.LayerNorm(ld),
        )

        self.flatten = nn.Flatten()
        
        self.final_layer = nn.Sequential(
            nn.Linear(ld, ld),
            nn.LeakyReLU(),
            nn.Linear(ld, 66),
            nn.LeakyReLU(),
            nn.Linear(66, 66),
        )   
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor):
        z, lengths, mu, logvar = self.encode(src)
        recon = self.decode(z, lengths)
        return recon, z, mu, logvar

    def encode(self, src):
        bs, seq_len = src.shape[0], src.shape[1]
        src = src.view(bs, seq_len, 66)
        # get lengths
        lengths = torch.tensor([len(seq) for seq in src], dtype=torch.float32).to(src.device)

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

        x = x[:, :2]  # select first frames
        x = self.flatten(x)
        x = self.latent_enc_linear(x)

        mu, logvar = x[:, :self.latent_dim], x[:, self.latent_dim:]
        z = self.reparametrize(mu, logvar)  # resample
        if self.verbose: print('latent:', z.shape)

        latentdim = torch.prod(torch.tensor(z.shape[1:]))
        if self.verbose: print('latentdim:', latentdim)

        return z, lengths, mu, logvar
    
    def reparametrize(self, mu, logvar):
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        return z

    
    def decode(self, z: Tensor, lengths: List[int]):
        if self.verbose: print('DECODING')
        mask = lengths_to_mask(lengths, z.device, self.seq_len)
        bs, nframes = mask.shape
        if self.verbose: 
            print('batch size:', bs, 'nframes:', nframes, 'z shape:', z.shape)
            print('mask:', mask)

        # ask transformer to generate the rest of the sequence
        
        z = z.unsqueeze(1)
        if self.verbose: print('z:', z.shape)

        # apply transformer
        queries = torch.zeros(bs, nframes, self.latent_dim, device=z.device)
        if self.verbose: print('queries:', queries.shape)
        queries = self.query_pos_decoder(queries)
        queries = queries.repeat(bs, 1, 1)

        if self.verbose: 
            print('queries:', queries.shape)
            print('mask:', mask.shape)
            print('z:', z.shape)

        z = self.trans_dec(tgt=queries,
                           memory=z,
                            tgt_key_padding_mask=~mask)
        if self.verbose: print('skip trans dec2:', z.shape)

        if self.verbose: print('final layer:', z)
        
        # final layer
        output = self.final_layer(z)
        if self.verbose: print('final layer:', output.shape)
        output[~mask] = 0
        feats = output#.permute(1, 0, 2)
        if self.verbose: print('feats:', feats.shape)

        feats = feats.view(bs, nframes, 22, 3)
        # print('final_output:', feats.shape)
        return feats


# Lightning Module
class MotionVAE(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        verbose: bool = False,
        **kwargs,
    ):
        super(MotionVAE, self).__init__()

        self.lr = kwargs.get("learning_rate")
        self.clip = kwargs.get("clip_grad_norm", 1)
        self.save_animations_freq = kwargs.get("save_animations_freq", -1)
        self.epochs_animated = []

        self.model = {
            "VAE1": VAE1,
            "VAE4": VAE4,
            "VAE5": VAE5,
            "VAE6": VAE6,
        }[model_name](**kwargs)
        assert self.model is not None, f"Model {model_name} not found"

        if verbose: self.model.verbose = True

        loss_weights = kwargs.get("LOSS")
        self.loss_function = VAE_Loss(loss_weights)

        self.val_outputs = {}
        
    def forward(self, x):
        return self.model(x)
    
    def encode(self, x):
        z = self.model.encode(x)[0]  # only return z
        return z

    def decode(self, z):
        return self.model.decode(z)


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
        try:
            
            plot_3d_motion_animation(
                recon_seq, "recon", figsize=(10, 10), fps=20, radius=2, save_path=fname, velocity=False,)
            plt.close()
            shutil.copyfile(fname, f"{self.folder}/recon_latest.mp4")  # copy file to latest
        except Exception as e:
            print(f"Error: {e}")

        

        if self.current_epoch == 0:
            plot_3d_motion_animation(
                motion_seq, "true", figsize=(10, 10), fps=20, radius=2, save_path=f"{self.folder}/recon_true.mp4", velocity=False)
            plt.close()

    def decompose_recon(self, motion_seq):
        pose0 = motion_seq[:, :1]
        root_travel = motion_seq[:, :, :1, :]
        root_travel = root_travel - root_travel[:1]  # relative to the first frame
        motion_less_root = motion_seq - root_travel  # relative motion
        velocity = torch.diff(motion_seq, dim=1)
        velocity_relative = torch.diff(motion_less_root, dim=1)

        return pose0, velocity_relative, root_travel, motion_less_root, velocity

    def _common_step(self, batch, batch_idx):
        motion_seq, text, action_group, action = batch
        recon, z, mu, logvar = self(motion_seq)
        
        pose0_pred, vel_rel_pred, root_trvl_pred, motion_rel_pred, vel_pred = self.decompose_recon(recon)
        pose0_gt, vel_rel_gt, root_trvl_gt, motion_rel_gt, vel_gt = self.decompose_recon(motion_seq)
        
        loss_data = {
            "VELOCITYRELATIVE_L2": {
                "true": vel_rel_gt,
                "rec": vel_rel_pred,
            },
            'VELOCITY_L2': {
                'true': vel_gt,
                'rec': vel_pred
            },

            "ROOT_L2": {
                "true": root_trvl_gt,
                "rec": root_trvl_pred,
            },  
            "POSE0_L2": {
                "true": pose0_gt,
                "rec": pose0_pred,
            }, 
            "MOTION_L2": {
                "true": motion_seq,
                "rec": recon,
            }, 
            "MOTIONRELATIVE_L2": {
                "true": motion_rel_gt,
                "rec": motion_rel_pred,
            }, 
            'DIVERGENCE_KL': {'mu': mu, 'logvar': logvar}
        }

        total_loss, losses_scaled, losses_unscaled = self.loss_function(loss_data)
        return dict(
            total_loss=total_loss,
            losses_scaled=losses_scaled,
            losses_unscaled=losses_unscaled,
            motion_seq=motion_seq,
            recon_seq=recon,
            text=text,
        )

    def training_step(self, batch, batch_idx):
        res = self._common_step(batch, batch_idx)
        
        loss = {k + "_trn": v for k, v in res["losses_unscaled"].items()}

        self.log_dict(loss, prog_bar=True, logger=True)
        self.log("total_loss_trn", res["total_loss"])
        # clip gradients --> do i do this here? # TODO
        if self.clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
        return res["total_loss"]

    def validation_step(self, batch, batch_idx):
        res = self._common_step(batch, batch_idx)
        # loss = {k + "_val": v for k, v in res["losses_unscaled"].items()}
        # self.log_dict(loss)
        self.log("total_loss_val", res["total_loss"])
        
        self.val_outputs['motion_seq'] = res['motion_seq'][0]
        self.val_outputs['recon_seq'] = res['recon_seq'][0]

    def on_validation_epoch_end(self):
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
        res = self._common_step(batch, batch_idx)
        loss = {k + "_tst": v for k, v in res["losses_unscaled"].items()}
        # self.log("test_loss", loss)
        # we want to add test loss final to the tensorboard
        self.log_dict(loss)

        return loss


    def configure_optimizers(self):
        # configure optimizers and schedulers
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'total_loss_val',
            }
        }
    


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="VAE4")
    args = args.parse_args()

    model = MotionVAE(model_name=args.model_name, verbose=True)

    x = torch.randn(128, 160, 22, 3)
    mu, logvar, z, recon = model(x)
    print('x shape: ', x.shape)
    print('recon shape: ', recon.shape)
    print('mu shape: ', mu.shape)
    print('logvar shape: ', logvar.shape)
    print('z shape: ', z.shape)
