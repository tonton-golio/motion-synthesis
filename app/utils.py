import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math

def load_or_save_fig(savepath, deactivate=False, darkmode=False):
    """
    Decorator to load an image if it exists, otherwise create and save it.

    Parameters:
    - savepath: Path to save/load the image

    Returns:
    - Wrapper function
    """
    if darkmode:
        savepath = savepath.replace('.png', '_darkmode.png')
    def decorator(func):
        def wrapper(*args, **kwargs):
            if deactivate:
                return func(*args, **kwargs)
            if os.path.exists(savepath):
                im = plt.imread(savepath)
                fig = plt.figure()
                plt.imshow(im)
                plt.axis('off')
                if darkmode:
                    fig.set_facecolor('black')
                plt.tight_layout()
                return fig
            else:
                fig = func(*args, **kwargs)
                fig.savefig(savepath, bbox_inches='tight')
                return fig
        return wrapper
    return decorator


def kl_divergence(mu, logvar):
    """
    Calculate the KL divergence loss, encouraging a more compact latent space by penalizing large values of mu and sigma.
    
    Parameters:
        mu (Tensor): The mean vector of the latent space distribution.
        logvar (Tensor): The log variance vector of the latent space distribution.
    
    Returns:
        Tensor: The computed KL divergence loss.
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def kl_score(x):
    """
    Calculate the KL score, which is the KL divergence between the input tensor and the standard normal distribution.
    
    Parameters:
        x (Tensor): The input tensor.
    
    Returns:
        Tensor: The computed KL score.
    """
    mu = x.mean()
    logvar = torch.log(x.var())
    return kl_divergence(mu, logvar)

# TODO: make a wrapper which makes a function darkmode

class VarianceSchedule(nn.Module):

    def __init__(self, timesteps, method="cosine", **kwargs):
        super(VarianceSchedule, self).__init__()
        self.timesteps = timesteps

        if method == "cosine":
            # st.write('using cosine, with epsilon:', kwargs.get("epsilon", 0.008))
            betas = self._cosine_variance_schedule(timesteps, epsilon=kwargs.get("epsilon", 0.008))
        elif method == "linear":
            betas = self._linear_variance_schedule(timesteps, 
                                                beta_start=kwargs.get("beta_start", 1e-5),
                                                beta_end=kwargs.get("beta_end", .01))
        elif method == "square":
            betas = self._sqr_variance_schedule(timesteps, 
                                                beta_start=kwargs.get("beta_start", 1e-4),
                                                beta_end=kwargs.get("beta_end", .1))
        else:
            raise NotImplementedError

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=-1)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def _cosine_variance_schedule(self, timesteps, epsilon=0.08):
        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        f_t = (
            torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5)
            ** 2
        )
        betas = torch.clip(1 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)
        # betas = betas / torch.max(betas)
        return betas
    
    def _linear_variance_schedule(self, timesteps, beta_start=1e-4, beta_end=0.02):
        betas = torch.linspace(beta_start, beta_end, steps=timesteps + 1, dtype=torch.float32)
        betas = torch.clip(betas[1:] , 0.0, 0.999)
        return betas
    
    def _sqr_variance_schedule(self, timesteps, beta_start=1e-4, beta_end=0.02):
        steps = torch.linspace(beta_start**0.5, beta_end**0.5, steps=timesteps + 1, dtype=torch.float32)
        f_t = steps**2
        betas = torch.clip(f_t[1:] , 0.0, 0.999)
        return betas


    def forward(self, x, t, noise=None, clip=False, multiplier=0.25, noise_type='normal'):
        A, B = self.alphas_cumprod[t], self.sqrt_one_minus_alphas_cumprod[t]
        if noise_type == 'normal':
            noise = torch.randn_like(x) if noise is None else noise
        elif noise_type == 'uniform':
            noise = (torch.rand_like(x) - 0.5)*2 if noise is None else noise
        # x_t = self.sqrt_alphas_cumprod[t] * x + self.sqrt_one_minus_alphas_cumprod[t] * noise
        x_t = A * x + B * noise
        if clip:
            x_t = torch.clip(x_t, 0.0, 1.0)
        return x_t


def print_scientific(x):
    return "{:.2e}".format(x)

activation_dict = {
    # soft step
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'softsign': nn.Softsign(),

    # rectifiers
    'leaky_relu': nn.LeakyReLU(),
    'ReLU': nn.ReLU(),
    'elu': nn.ELU(),
    'swish': nn.SiLU(),

    # identity
    'None': nn.Identity(),
    None: nn.Identity(),
}

def plotUMAP(latent, labels, latent_dim, KL_weight,  save_path, show=False, max_points=5000):
    import umap
    
    if latent.shape[0] > max_points:
        idx = torch.randperm(latent.shape[0])[:max_points]
        latent = latent[idx]
        labels = labels[idx]

    reducer = umap.UMAP()
    projection = reducer.fit_transform(latent.cpu().detach().numpy())
    
    fig = plt.figure()
    plt.scatter(projection[:, 0], projection[:, 1], c=labels.cpu().numpy(), cmap='tab10', alpha=0.5, s=4)
    plt.colorbar()
    plt.title(f'UMAP projection of latent space (LD={latent_dim}, KL={print_scientific(KL_weight)})')
    
    if save_path is not None:
        plt.savefig(f'{save_path}/projection_LD{latent_dim}_KL{print_scientific(KL_weight)}.png')
    
        return projection, reducer
    elif show:
        plt.show()
    return fig
