import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
import torch.nn.functional as F
from utils_pose import plot_3d_motion_frames_multiple
from modules.loss import VAE_Loss

# LINEAR VAE
class LinearLayer(nn.Module):
    """
    Linear layer with batchnorm and dropout.
    """
    def __init__(self, input_dim, output_dim, activation=F.relu, dropout=0.01, batch_norm=True):
        super(LinearLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.activation(self.fc(x))
        x = F.dropout(x, self.dropout)
        if self.batch_norm:
            x = self.bn(x)
        return x

class LinearVAE(nn.Module):
    def __init__(self, 
                 input_dim=66, hidden_dims=[128, 256, 128],  dropout=0.01, latent_dim=32, activation=F.relu, **kwargs):
        super(LinearVAE, self).__init__()
        self.input_dim = input_dim
        self.dropout = dropout

        self.activation = activation

        self.enc = nn.Sequential(*(LinearLayer(in_, out_, dropout=dropout) 
                                   for in_, out_ in zip([input_dim]+hidden_dims[:-1], hidden_dims)))
        self.enc_final = nn.Linear(hidden_dims[-1], latent_dim*2)

        self.dec = nn.Sequential(*(LinearLayer(in_, out_, dropout=dropout) 
                                   for in_, out_ in zip([latent_dim]+hidden_dims[::-1], hidden_dims[::-1])))
        self.dec_final = nn.Linear(hidden_dims[0], input_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.enc(x)
        mu, logvar = self.enc_final(x).chunk(2, dim=1)
        z = self.reparametrize(mu, logvar)
        x = self.decode(z)
        return x, z, mu, logvar

    def encode(self, x):
        x = x.view(-1, self.input_dim)
        x = self.enc(x)
        mu, logvar = self.enc_final(x)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.dec(z)
        x = self.dec_final(x)
        return x.view(-1, 22, 3)

# CONV VAE
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=F.relu, batch_norm=True):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = activation
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        if self.batch_norm:
            x = self.bn(x)
        return x
    
class ConvVAE(nn.Module):
    # this model should use linear layers to expand the input making a 2d field we can convolve

    def __init__(self,
                    input_dim=66, 
                    dim_mults=[1, 2, 3], 
                    latent_dim=32, 
                    dropout=0.01,
                    activation=F.relu,
                    **kwargs):
        super(ConvVAE, self).__init__()
        self.verbose = kwargs.get("verbose")
        self.input_dim = input_dim
        self.activation = activation
        
        dims = [input_dim] + [input_dim * m for m in dim_mults]
        self.linear_reorder = nn.Sequential(*(LinearLayer(in_, out_, dropout=dropout) 
                                for in_, out_ in zip(dims[:-1], dims[1:])))

        # encoder
        self.conv1 = ConvLayer(1, 8, kernel_size=3, stride=1, padding=0)
        self.conv2 = ConvLayer(8, 16, kernel_size=3, stride=2, padding=0)
        self.conv3 = ConvLayer(16, 4, kernel_size=3, stride=2, padding=0)

        # flatten
        self.flatten = nn.Flatten()

        self.fc_enc = nn.Linear(60, latent_dim*2)

        # decoder
        dims = [latent_dim] + [input_dim * m for m in dim_mults[::-1]] + [input_dim]
        self.fc_dec = nn.Sequential(*(LinearLayer(in_, out_, dropout=dropout) 
                                for in_, out_ in zip(dims[:-1], dims[1:])))
        
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x = self.decode(z)
        return x, z, mu, logvar

    def encode(self, x):
        bs = x.size(0)
        x = x.view(bs, self.input_dim)
        if self.verbose: print('before linaer', x.shape)
        x = self.linear_reorder(x)
        x = x.view(bs, 1, -1)
        if self.verbose: print('before conv1', x.shape)
        x = self.conv1(x)
        if self.verbose: print('before conv2', x.shape)
        x = self.conv2(x)
        if self.verbose: print('before conv3', x.shape)
        x = self.conv3(x)
        x = self.flatten(x)
        if self.verbose: print('before', x.shape)
        x = self.fc_enc(x)
        if self.verbose: print('before', x.shape)
        mu, logvar = torch.chunk(x, 2, dim=1)
        z = self.reparametrize(mu, logvar)

        return z, mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_dec(z)
        x = x.view(-1, 22, 3)
        return x

# GRAPH VAE (graph data so not used)
class GraphPoseAutoencoder(nn.Module):
    def __init__(self, 
                    shape=(22, 3), 
                    hidden_dims=[66, 128, 256, 512, 1024],
                    latent_dim=56, 
                    ):
        super(GraphPoseAutoencoder, self).__init__()
        self.shape = shape
        self.hidden_dims = hidden_dims

        self.latent_dim = latent_dim

        # ENCODER
        self_encoder_conv = nn.Sequential(
            GCNConv(3, 32),
            GCNConv(32, 64),
            GCNConv(64, 32)
        )

        self_encoder_linear_block = nn.Sequential(
            *(nn.Linear(in_, out_) for in_, out_ in zip([32*shape[0], *hidden_dims[:-1]], hidden_dims))
        )

        self.fc_enc_out_mu = nn.Linear(hidden_dims[-1], self.latent_dim)
        self.fc_enc_out_logvar = nn.Linear(hidden_dims[-1], self.latent_dim)

        # DECODER
        self_decoder_linear_block = nn.Sequential(
            *(nn.Linear(in_, out_) for in_, out_ in zip([self.latent_dim, *hidden_dims[::-1]], hidden_dims[::-1]))
        )

        self.fc_dec_out = nn.Linear(hidden_dims[0], 32*shape[0])

    def encode(self, x, edge_index):
        x = self.conv_enc_1(x, edge_index)

        # x = self.conv_enc_2(x, edge_index)
        # x = F.relu(x)
        # x = self.conv_enc_3(x, edge_index)
        # x = F.relu(x)
        # if self.verbose: print('x:', x.shape)
        # x = self.conv_enc_4(x, edge_index)
        # x = F.relu(x)

        x = x.view(-1, 32*22)

        x = F.relu(self.fc_enc_1(x))

        mu = self.fc_enc_out_mu(x)
        logvar = self.fc_enc_out_logvar(x)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, x, edge_index):
        x = F.relu(self.fc_dec_1(x))
        x = F.relu(self.fc_dec_2(x))
        x = x.view(-1, 32)
        # x = self.conv_dec_1(x, edge_index)
        # if self.verbose: print('x:', x.shape)
        # x = F.relu(x)
        
        # x = self.conv_dec_2(x, edge_index)
        # if self.verbose: print('x:', x.shape)
        # x = F.relu(x)
        # x = self.conv_dec_3(x, edge_index)
        # x = F.relu(x)
        x = self.conv_dec_4(x, edge_index)
        return x

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        mu, logvar = self.encode(x, edge_index)
        z = self.reparametrize(mu, logvar)
        x = self.decode(z, edge_index)
        return x, z,  mu, logvar

# inspired by: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/06-graph-neural-networks.html#
# GRAPH VAE (matrix data)
def get_adjacency_matrix():
    kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]

    edge_index = []
    for chain in kinematic_chain:
        for i in range(len(chain)-1):
            edge_index.append([chain[i], chain[i+1]])
            edge_index.append([chain[i+1], chain[i]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    adjacency_matrix = np.zeros((22, 22))
    for chain in kinematic_chain:
        for i in range(len(chain) - 1):
            adjacency_matrix[chain[i], chain[i+1]] = 1
            adjacency_matrix[chain[i+1], chain[i]] = 1

    adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32).unsqueeze(0)
    return adjacency_matrix

class GCNLayer(nn.Module):
    def __init__(self, c_in, c_out, **kwargs):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out)
        self.adjacency_matrix = get_adjacency_matrix()



    def forward(self, node_feats):
        """Forward.

        Args:
            node_feats: Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix: Batch of adjacency matrices of the graph. If there is an edge from i to j,
                         adj_matrix[b,i,j]=1 else 0. Supports directed edges by non-symmetric matrices.
                         Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        adj_matrix = self.adjacency_matrix.to(node_feats.device)
        # stack adj matrix for each sample in the batch
        adj_matrix = adj_matrix.repeat(node_feats.size(0), 1, 1)
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
        node_feats = self.projection(node_feats)
        node_feats = torch.bmm(adj_matrix, node_feats)
        node_feats = node_feats / num_neighbours
        return node_feats

class GnnVAE(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        c_out,
        num_layers=2,
        latent_dim=2,
        layer_name="GCN",
        n_lin_layers=2,
        n_lin_units=64,
        dp_rate=0.1,
        **kwargs,
    ):
        """GNNModel.

        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of the output features. Usually number of classes in classification
            num_layers: Number of "hidden" graph layers
            layer_name: String of the graph layer to use
            dp_rate: Dropout rate to apply throughout the network
            kwargs: Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        self.c_out = c_out
        # Encoder
        layers_graphconv = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            layers_graphconv += [
                GCNLayer(c_in=in_channels, c_out=out_channels, **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate),
            ]
            in_channels = c_hidden
        layers_graphconv += [GCNLayer(c_in=in_channels, c_out=c_out, **kwargs),
                             nn.ReLU(inplace=True), 
                             nn.Dropout(dp_rate)]
        self.layers_graphconv = nn.ModuleList(layers_graphconv)

        layers_lin_enc = []
        layers_lin_enc += [nn.Linear(c_out*22, n_lin_units)]
        for l_idx in range(n_lin_layers - 1):
            layers_lin_enc += [
                nn.Linear(n_lin_units, n_lin_units),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate),
            ]
        layers_lin_enc += [nn.Linear(n_lin_units, latent_dim*2)]

        self.layers_lin_enc = nn.ModuleList(layers_lin_enc)

        
        # Latent space
        self.layers_mu = nn.Linear(latent_dim*2, latent_dim)
        self.layers_logvar = nn.Linear(latent_dim*2, latent_dim)
        # self.antiembed = nn.Linear(latent_dim, latent_dim*2)
        
        # Decoder
        layers_antiembed = [
            nn.Linear(latent_dim, latent_dim*2),
            nn.ReLU(inplace=True),
            nn.Dropout(dp_rate),
            nn.Linear(latent_dim*2, n_lin_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dp_rate),
        ]

        for l_idx in range(n_lin_layers - 1):
            layers_antiembed += [
                nn.Linear(n_lin_units, n_lin_units),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate),
            ]
        layers_antiembed += [nn.Linear(n_lin_units, c_out*22)]
        self.antiembed = nn.Sequential(*layers_antiembed)

    
        layers_dec = []
        in_channels, out_channels = c_out, c_hidden
        
        for l_idx in range(num_layers - 1):
            layers_dec += [
                GCNLayer(c_in=in_channels, c_out=out_channels, **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate),
            ]
            in_channels = c_hidden
        layers_dec += [GCNLayer(c_in=in_channels, c_out=c_in, **kwargs)]
        self.layers_dec = nn.ModuleList(layers_dec)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        for layer in self.layers_graphconv:
            x = layer(x)
        x = x.view(x.size(0), -1)
        for layer in self.layers_lin_enc:
            x = layer(x)
        mu, logvar = torch.chunk(x, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        x = self.antiembed(z)
        x = x.view(x.size(0), 22, self.c_out)
        for layer in self.layers_dec:
            x = layer(x)
        return x

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x = self.decode(z)
        return x, z, mu, logvar


# Lightning Module
class PoseVAE(pl.LightningModule):
    def __init__(self, model_name, **kwargs):
        super().__init__()

        if model_name == "LINEAR":  self.model = LinearVAE(**kwargs)
        elif model_name == "CONV":  self.model = ConvVAE(**kwargs)
        elif model_name == "GRAPH": self.model = GnnVAE(**kwargs)
            
        self.loss = VAE_Loss(kwargs.get("LOSS"))
        self.optimizer = kwargs.get("optimizer")
        self.lr = kwargs.get("learning_rate")

        self.test_losses = []

    def forward(self, batch, stage='train'):
        x = batch
        recon, z, mu, logvar = self.model(x)
        loss_data = {
            'MSE_L2' : {
                'true': x,
                'rec': recon,
            },
            'DIVERGENCE_KL': {'mu': mu, 'logvar': logvar}
        }
        total_loss, losses_scaled, losses_unscaled = self.loss(loss_data)
        losses_scaled = {f"{k}_{stage}": v for k, v in losses_unscaled.items()}
        losses_unscaled = {f"{k}_{stage}": v for k, v in losses_unscaled.items()}

        
        return dict(total_loss=total_loss, losses_scaled=losses_scaled, losses_unscaled=losses_unscaled, recon=recon, x=x, z=z, mu=mu, logvar=logvar)
    
    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        output = self(batch)
        # loss = {k + "_train": v for k, v in output['losses_unscaled'].items()}
        self.log_dict(output['losses_unscaled'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('total_train', output['total_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return output['total_loss']
    
    def validation_step(self, batch, batch_idx):
        output = self(batch, stage='val')
        self.log('val_loss', output['total_loss'], on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        if batch_idx == 0:
            recon = output['recon']
            x = output['x']
            z = output['z']
            mu = output['mu']
            logvar = output['logvar']

            batch_select = x[:6].detach().cpu().numpy()
            recon_select = recon[:6].detach().cpu().numpy()

            grid = plot_3d_motion_frames_multiple( [batch_select, recon_select],  ['gt', 'recon'], nframes=5, radius=2, figsize=(20, 8), return_array=True)

            self.logger.experiment.add_image("input", grid, global_step=self.global_step)

    def test_step(self, batch, batch_idx):
        output = self(batch, stage='test')
        self.log('test_loss', output['total_loss'], on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.test_losses.append(output['total_loss'])
