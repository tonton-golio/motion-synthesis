import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

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



# TODO: make a wrapper which makes a function darkmode


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
