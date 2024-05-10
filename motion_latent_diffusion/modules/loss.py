import torch
from torch import nn
from torch.nn import functional as F

class BatchSequenceMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')  # Compute loss per element

    def forward(self, preds, trues, lengths):
        # preds = torch.tensor(preds, dtype=torch.float32)
        # trues = torch.tensor(trues, dtype=torch.float32)
        
        losses = self.criterion(preds, trues).sum(dim=2)  # Compute loss per element
        mask = torch.arange(losses.size(1)).expand(len(lengths), losses.size(1)).to(preds.device) < lengths.unsqueeze(1)
        # print(mask.shape, losses.shape, lengths.shape)

        masked_losses = losses * mask
        # print(masked_losses.shape)
        # print(lengths.shape, lengths)
        loss_per_sequence = masked_losses.sum(dim=1) / lengths
        return loss_per_sequence.sum()

class VAE_Loss(nn.Module):
    def __init__(self, loss_weights):
        super(VAE_Loss, self).__init__()
        self.loss_weights = loss_weights
        self.sequence_mse_loss = BatchSequenceMSELoss()  # integrate the sequence MSE loss here

    def forward(self, loss_data, lengths=None):
        total_loss = 0.0
        losses_unscaled = {}
        losses_scaled = {}
        for k, v in loss_data.items():
            name, method = k.split('_')
            weight = self.loss_weights.get(k, 0)
            if weight == 0:
                continue

            if method == 'L2':
                if lengths is not None:
                    l = self.sequence_mse_loss(v['rec'], v['true'], lengths)# if lengths else F.mse_loss(v['rec'], v['true'], reduction='sum')
                    losses_unscaled[k] = l / lengths.sum()# if lengths else F.mse_loss(v['rec'], v['true'], reduction='mean')
                else:
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
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


if __name__ == '__main__':
    # Example of setting up the VAE_Loss
    loss_weights = {
        'reconstruction_L2': 1.0,
        'kl_KL': 0.5  # Example weights
    }

    vae_loss = VAE_Loss(loss_weights)

    # Example data
    loss_data = {
        'reconstruction_L2': {'rec': [[0.9, 2.1, 3.1, 4.1, 0, 0, 0], [0.9, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1]],
                            'true': [[1, 2, 3, 4, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7]]},
        'kl_KL': {'mu': torch.zeros(10, 32), 'logvar': torch.zeros(10, 32)}
    }

    lengths = torch.tensor([4, 7])

    # Calculate loss
    total_loss, losses_scaled, losses_unscaled = vae_loss(loss_data, lengths)
    print('Total Loss:', total_loss.item())
