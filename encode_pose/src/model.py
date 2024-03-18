import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv
from torch.utils.data import DataLoader

# lightning module
import pytorch_lightning as pl


class linearblock(pl.LightningModule):
    def __init__(self, input_dim, output_dim, activation=F.relu, dropout=0.01, batch_norm=True):
        super(linearblock, self).__init__()
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

class LinearPoseAutoencoder(pl.LightningModule):
    def __init__(self, input_dim=66, hidden_dims=[66, 128, 256, 512, 1024],  dropout=0.01, latent_dim=32, activation=F.relu):
        super(LinearPoseAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.dropout = dropout

        self.activation = activation

        self.enc = nn.Sequential(
            *(linearblock(in_, out_, dropout=dropout) for in_, out_ in zip([input_dim]+hidden_dims[:-1], hidden_dims)),
        )
        self.enc_final = nn.Linear(hidden_dims[-1], latent_dim)

        self.dec = nn.Sequential(
            *(linearblock(in_, out_, dropout=dropout) for in_, out_ in zip([latent_dim]+hidden_dims[::-1], hidden_dims[::-1]))
        )
        self.dec_final = nn.Linear(hidden_dims[0], input_dim)

    def forward(self, x):
        # print(x.shape)

        x = x.view(-1, self.input_dim)

        x = self.enc(x)
        x = self.enc_final(x)

        x = self.dec(x)
        x = self.dec_final(x)

        return x.view(-1, 22, 3)
    
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
        return mu, logvar
    
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
        return x, mu, logvar


import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
import torch_geometric.nn as geom_nn
import torchvision
from utils import plot_3d_motion_frames_multiple


# inspired by: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/06-graph-neural-networks.html#

def get_loss_function(loss_function, kl_weight=0.1, **kwargs):
    if loss_function == 'L1Loss':
        return nn.L1Loss()
    elif loss_function == 'MSELoss':
        return nn.MSELoss()
    elif loss_function == 'SmoothL1Loss':
        return nn.SmoothL1Loss()
    
    elif loss_function == "MSELoss + KL":
        # this should be the sum of the reconstruction loss and the KL divergence
        mse = nn.MSELoss()
        kl = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return lambda x, y, mu, logvar: mse(x, y) + kl_weight * kl(mu, logvar)

    
    raise ValueError(f'Loss function {loss_function} not found')

def get_optimizer(model, optimizer_name, learning_rate):
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError('Invalid optimizer')
    return optimizer

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

class GnnAutoEncoder(nn.Module):
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

    def forward(self, x):
        """Forward.

        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        # print(x.shape)
        for layer in self.layers_graphconv:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            x = layer(x)
            # print(x.shape)
        x = x.view(x.size(0), -1)
        # print('x:', x.shape)
        for layer in self.layers_lin_enc:
            x = layer(x)
            # print(x.shape)
        mu = self.layers_mu(x)
        # print('mu:', mu.shape)
        logvar = self.layers_logvar(x)
        # print('logvar:', logvar.shape)
        z = self.reparameterize(mu, logvar)
        
        # print('Reparameterize done', z.shape)
        x = self.antiembed(z)
        x = x.view(x.size(0), 22, self.c_out)
        # print('Reparameterize done')
        for layer in self.layers_dec:
            # print('Layer:', layer, 'x:', x.shape)
            x = layer(x)

        return x , mu, logvar

class MLPAutoencoder(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers=2, dp_rate=0.1):
        """MLPAutoencoder.

        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of the output features. Usually number of classes in classification
            num_layers: Number of hidden layers
            dp_rate: Dropout rate to apply throughout the network
        """
        super().__init__()
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            layers += [nn.Linear(in_channels, out_channels), nn.ReLU(inplace=True), nn.Dropout(dp_rate)]
            in_channels = c_hidden
        layers += [nn.Linear(in_channels, c_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        """Forward.

        Args:
            x: Input features per node
        """
        return self.layers(x)

class NodeLevelGNNAutoencoder(L.LightningModule):
    def __init__(self, model_name, loss_function,kl_weight, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        # self.save_hyperparameters()

        if model_name == "MLP":
            self.model = MLPAutoencoder(**model_kwargs)
        else:
            self.model = GnnAutoEncoder(**model_kwargs)


        self.loss_function = get_loss_function(loss_function, kl_weight=kl_weight)
        self.optimizer = model_kwargs.get("optimizer", "Adam")
        self.lr = model_kwargs.get("lr", 1e-3)

    def forward(self, data, mode="train"):
        recon, mu, logvar = self.model(data)
        loss = self.loss_function(recon, data, mu, logvar)
        return loss, recon

    def configure_optimizers(self):
        # this is also where we would put the scheduler
        return get_optimizer(self, self.optimizer, self.lr)

    def training_step(self, batch, batch_idx):
        loss, recon = self.forward(batch, mode="train")
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if batch_idx % 999999999 == 0:
            """
            Here we're loggig a row of images. 
            """
            print('Logging images')
            
            self.logger.experiment.add_graph(self.model, batch)
            
            # grid = torchvision.utils.make_grid(batch)
            batch_select = batch[:6].detach().cpu().numpy()
            recon_select = recon[:6].detach().cpu().numpy()

            # subtract the mean
            # batch_select -= batch_select[:,0:1]
            # recon_select -= recon_select[:,0:1]

            grid = plot_3d_motion_frames_multiple(
                [batch_select, recon_select], 
                ['gt', 'recon'], nframes=5, radius=2, figsize=(20, 8), return_array=True)
            self.logger.experiment.add_image("input", grid, global_step=self.global_step)
            


        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon = self.forward(batch, mode="val")
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss, recon = self.forward(batch, mode="test")
        self.log("test_loss", loss)
