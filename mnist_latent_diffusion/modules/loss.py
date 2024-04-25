import torch
from torch import nn
from torch.nn import functional as F

class VAE_Loss(nn.Module):
    def __init__(self, loss_weights):
        """
        Initialize the CustomLoss module with a structure for loss_weights that defines
        each loss component, its calculation method, and its weight.

        Parameters:
            loss_weights (dict): A dictionary where each key is 'name_method', and the value is weight
        """
        super(VAE_Loss, self).__init__()
        self.loss_weights = loss_weights
           
    def forward(self, loss_data):
        """
        Calculate and return the custom loss based on the provided loss data and methods defined
        in the initialization.

        Parameters:
            loss_data (dict): A dictionary with keys for each loss component (including 'mu' and 'logvar'
                              for KL divergence) and values containing the data necessary for loss calculation.

        Returns:
            float: The total loss calculated from the sum of all components.
            dict: A dictionary containing the calculated losses for each component and the total loss.
            dict: loss unscaled
        """

        total_loss = 0.0
        losses_unscaled = {}
        losses_scaled  = {}
        for k, v in loss_data.items():
            # print(k)
            name, method = k.split('_')
            weight = self.loss_weights.get(k)
            # print(weight)
            if weight in [0, None]:
                continue

            if method == 'L2':
                losses_unscaled[k] = F.mse_loss(v['rec'], v['true'], reduction='mean')
            elif method == 'L1':
                losses_unscaled[k] = F.l1_loss(v['rec'], v['true'], reduction='mean') 
            elif method == 'KL':
                losses_unscaled[k] = self.kl_divergence(v['mu'], v['logvar'])
            elif method == 'BCE':
                losses_unscaled[k] = F.binary_cross_entropy(v['rec'], v['true'], reduction='mean')
            else:
                raise ValueError(f"Invalid loss method '{method}' provided for loss component '{k}'.")

            losses_scaled[k] = weight * losses_unscaled[k]
            total_loss += losses_scaled[k]

        return total_loss, losses_scaled, losses_unscaled

    def kl_divergence(self, mu, logvar):
        """
        Calculate the KL divergence loss, encouraging a more compact latent space by penalizing large values of mu and sigma.
        
        Parameters:
            mu (Tensor): The mean vector of the latent space distribution.
            logvar (Tensor): The log variance vector of the latent space distribution.
        
        Returns:
            Tensor: The computed KL divergence loss.
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

if __name__ == '__main__':

    # make toy loss_weights
    loss_weights = {'RECONSTRUCTION_L2': 1.0, 'DIVERGENCE_KL': 1e-06}

    criterion = VAE_Loss(loss_weights)

    x1 = torch.randn(1, 1, 28, 28)
    x2 = torch.randn(1, 1, 28, 28)
    mu = torch.randn(1, 20)
    logvar = torch.randn(1, 20)

    loss_data = {
        'RECONSTRUCTION_L2' : {'rec': x1, 'true': x2},
        'DIVERGENCE_KL' : {'mu': mu, 'logvar': logvar}
    }

    # Calculate the loss
    total_loss, losses_scaled, losses_unscaled = criterion(loss_data)

    print('total_loss:', total_loss)
    print('losses_scaled:', losses_scaled)
    print('losses_unscaled:', losses_unscaled)

