import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from utils import plot_3d_motion_animation, translate
except:
    from motion_latent_diffusion.utils import plot_3d_motion_animation, translate


class TimeMLP(nn.Module):
    '''
    naive introduce timestep information to feature maps with mlp
    '''
    def __init__(self,in_dim,hidden_dim,out_dim):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(in_dim,hidden_dim),
                                nn.SiLU(),
                               nn.Linear(hidden_dim,out_dim))
        self.act=nn.SiLU()

    def forward(self,t):
        t_emb=self.mlp(t)#.unsqueeze(-1).unsqueeze(-1)
        # print('t_emb', t_emb.shape)
        #x=x+t_emb
  
        return t_emb
    
class MLP(nn.Module):
    '''
    naive introduce timestep information to feature maps with mlp
    '''
    def __init__(self,in_dim,hidden_dim,out_dim, nhidden=5, activation=nn.SiLU(), dropout=0.1):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(in_dim,hidden_dim),
                                activation,
                               nn.Dropout(dropout))
        for _ in range(nhidden-1):
            self.mlp.add_module(f'hidden_{_}', nn.Linear(hidden_dim,hidden_dim))
            self.mlp.add_module(f'activation_{_}', activation)
            #self.mlp.add_module(f'dropout_{_}', nn.Dropout(dropout))
        # batch norm
        self.mlp.add_module('bn', nn.BatchNorm1d(hidden_dim))
        self.mlp.add_module('out', nn.Linear(hidden_dim,out_dim))
        
    def forward(self,x):
        x=self.mlp(x)
        return x
    
# class TextTransformer(nn.Module):
#     def __init__(self, target_embedding_dim=5, nhead=10, num_layers=3, dim_feedforward=512, dropout=0.1, activation='relu'):
#         super().__init__()
#         self.text_transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=target_embedding_dim,
#                 nhead=nhead,
#                 dim_feedforward=dim_feedforward,
#                 dropout=dropout,
#                 activation=activation,
#                 batch_first=True
#             ),
#             num_layers=num_layers,
#             norm=nn.LayerNorm(target_embedding_dim)
#         )

#     def forward(self, x):
#         return self.text_transformer(x)
    
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.register_buffer(
            "pe",
            self._get_sinusoidal_encoding(max_len, d_model),
        )

    def _get_sinusoidal_encoding(self, max_len, d_model):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return pe

    def forward(self, t):
        return self.pe[t]

class SimpleModel(nn.Module):
    # a simple model takes (batchsize, latent dim),
    # performs linear layers
    # and returns (batchsize, latent dim)

    def __init__(
        self,
        latent_dim,
        hidden_dim,
        nhidden=5,
        timesteps=1000,
        time_embedding_dim=64,
        dp_rate=0.1,
        verbose=False,
        **kwargs
    ):
        super(SimpleModel, self).__init__()
        self.verbose = verbose
        self.time_embedding_dim = time_embedding_dim
        self.target_embedding_dim = 512  # we use CLIP for target embedding
        
        self.dropout = dp_rate
        
        self.time_embedding = SinusoidalPositionalEmbedding(
            d_model=time_embedding_dim, max_len=timesteps
        )
        self.time_mlp = MLP(
            in_dim=time_embedding_dim,
            hidden_dim=time_embedding_dim*2,
            out_dim=latent_dim,
            nhidden=1,
            activation=nn.SiLU(),
            dropout=0.0)
            
        # self.time_mlp=TimeMLP(
        #     in_dim=time_embedding_dim, 
        #     hidden_dim=time_embedding_dim*2,
        #     out_dim=time_embedding_dim)
        
        self.target_mlp=MLP(
            in_dim=self.target_embedding_dim, 
            hidden_dim=self.target_embedding_dim*2,
            out_dim=latent_dim,
            nhidden=2,
            activation=nn.LeakyReLU(),
            dropout=dp_rate)
        
        # self.target_mlp=TimeMLP(
        #     in_dim=self.target_embedding_dim, 
        #     hidden_dim=self.target_embedding_dim*2,
        #     out_dim=latent_dim)
        
        self.latent_mlp=MLP(
            in_dim=latent_dim, 
            hidden_dim=latent_dim*2,
            out_dim=latent_dim,
            nhidden=1,
            activation=nn.LeakyReLU(),
            dropout=dp_rate)
        
        # self.latent_mlp=TimeMLP(
        #     in_dim=latent_dim, 
        #     hidden_dim=latent_dim*2,
        #     out_dim=latent_dim)
            
        self.fc1 = MLP(
            in_dim=latent_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            nhidden=nhidden,
            activation=nn.LeakyReLU(),
            dropout=dp_rate
        )

        self.fc2 = MLP(
            in_dim=latent_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            nhidden=nhidden,
            activation=nn.LeakyReLU(),
            dropout=dp_rate
        )

        self.fc3 = MLP(
            in_dim=latent_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            nhidden=nhidden,
            activation=nn.LeakyReLU(),
            dropout=dp_rate
        )

        # set initial weights

        # self.fc.apply(self._init_weights)
        # self.time_mlp.apply(self._init_weights)

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
            


    def forward(self, x, y, t):
        
        # start of forward
        if self.verbose:
            print()
            print('start of forward')
            print('x', x.shape, x.dtype)
            print('y', y.shape, y.dtype)
            print('t', t.shape, t.dtype)

        # time embedding
        t = self.time_embedding(t)  # (batchsize, time_embedding_dim)
        if self.verbose:
            print('t (after embedding)', t.shape)
        t = self.time_mlp(t)
        if self.verbose:
            print('t (after mlp)', t.shape)
        # target embedding
        y = self.target_mlp(y)
        if self.verbose:
            print('y (after mlp)', y.shape)
        # latent embedding
        x = self.latent_mlp(x)
        if self.verbose:
            print('x (after mlp)', x.shape)

        # sum
        x = x + y

        # fc1
        x = self.fc1(x)

        # fc2
        x = x + self.fc2(x + t)

        # fc3
        x = self.fc3(x)
        
        if self.verbose:
            print('x', x.shape)
        return x

class LatentDiffusionModel(nn.Module):
    def __init__(
        self,
        latent_dim=8,
        hidden_dim=64,
        nhidden=3,
        timesteps=1000,
        time_embedding_dim=64,
        target_embedding_dim=5,
        epsilon=0.008,
        dp_rate=0.1,
        verbose=False
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.in_channels = latent_dim

        betas = self._cosine_variance_schedule(timesteps, epsilon)
        # print('betas', betas.shape)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=-1)

        # print('alphas_cumprod', alphas_cumprod.shape)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        self.model = SimpleModel(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            nhidden=nhidden,
            timesteps=timesteps,
            time_embedding_dim=time_embedding_dim,
            target_embedding_dim=target_embedding_dim,
            dp_rate=dp_rate,
            verbose=verbose
        )

        self.rand_texts = []

    def forward(self, x, y, noise):
        # x:NCHW
        t = torch.randint(0, self.timesteps, (x.shape[0],)).to(x.device)
        # print('t from the LatentDIffusion forward', t)

        x_t = self._forward_diffusion(x, t, noise)

        # print('x_t', x_t.shape, )
        # print('t', t.shape)
        pred_noise = self.model(x_t, y, t)
        # print('pred_noise', pred_noise.shape)
        niose_added = x_t - x
        return pred_noise, niose_added

    @torch.no_grad()
    def sampling(self, texts, noise_mul, clipped_reverse_diffusion=True, device="cuda"):
        n_samples = len(texts)
        y = texts
        # print('y', y.shape, y.dtype, y)
        x_t = torch.randn(
            (n_samples, self.latent_dim)
        ).to(device)
        
        history = [x_t.clone()]

        for i in tqdm(range(self.timesteps - 1, -1, -1), desc="Sampling"):
            noise = torch.randn_like(x_t) * noise_mul
            t = torch.tensor([i for _ in range(n_samples)]).to(device)

            if clipped_reverse_diffusion:
                x_t = self._reverse_diffusion_with_clip(x_t, t, noise)
            else:
                x_t = self._reverse_diffusion(x_t, y, t, noise)

            history.append(x_t.clone())

        # x_t = (x_t + 1.0) / 2.0  # [-1,1] to [0,1]
        history = torch.stack(history, dim=1)
        return x_t, history

    def _cosine_variance_schedule(self, timesteps, epsilon=0.008):
        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        f_t = (
            torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5)
            ** 2
        )
        betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)

        return betas

    def _forward_diffusion(self, x_0, t, noise):
        # print('x_0', x_0.shape)
        # print('noise', noise.shape)
        # print('t', t)

        assert x_0.shape == noise.shape
        # print('self.sqrt_alphas_cumprod.gather(t)', self.sqrt_alphas_cumprod.gather(0,t).shape)
        # q(x_{t}|x_{t-1})

        A = self.sqrt_alphas_cumprod.gather(0, t).unsqueeze(1)
        B = self.sqrt_one_minus_alphas_cumprod.gather(0, t).unsqueeze(1)
        # print('A', A.shape)
        # print('B', B.shape)
        # print('noise', noise.shape)
        # print('x_0', x_0.shape)
        return A * x_0 + B * noise

    @torch.no_grad()
    def _reverse_diffusion(self, x_t, y, t, noise):
        """
        p(x_{t-1}|x_{t})-> mean,std

        pred_noise-> pred_mean and pred_std
        """

        x_t_copy = x_t.clone()
        pred_noise = self.model(x_t, y, t)
        return x_t_copy - pred_noise * t/self.timesteps  
        alpha_t = self.alphas[t][:, None]
        beta_t = self.betas[t][:, None]
        srqt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]

        # return x_t_copy - sqrt_one_minus_alpha_cumprod_t * pred_noise   

        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1)
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - ((1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred_noise
        )

        if t.float().max() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(
                x_t.shape[0], 1)
            std = torch.sqrt(
                beta_t * (1.0 - alpha_t_cumprod_prev) / (1.0 - alpha_t_cumprod)
            )
        else:
            std = 0.0

        # insert x_t_copy when t=0
        x_t = mean + std * noise
        x_t = torch.where(t == 0, x_t_copy, x_t)
        return x_t


        # x_t_copy = x_t.clone()
        # pred = self.model(x_t, y, t)

        # return (pred + x_t_copy) / 2.0

        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1)  # ,1,1)
        # print('alpha_t', alpha_t.shape)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(
            x_t.shape[0], 1
        )  # ,1,1)
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1)  # ,1,1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(
            -1, t
        ).reshape(
            x_t.shape[0], 1
        )  # ,1,1)
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - ((1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred
        )

        if t.float().min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(
                x_t.shape[0], 1
            )  # ,1,1)
            std = torch.sqrt(
                beta_t * (1.0 - alpha_t_cumprod_prev) / (1.0 - alpha_t_cumprod)
            )
        else:
            std = 0.0

        # insert x_t_copy when t=0
        x_t = mean + std * noise
        x_t = torch.where(t == 0, x_t_copy, x_t)
        return x_t
# make pl model
class MotionLatentDiffusion(pl.LightningModule):
    def __init__(
        self,
        decode,
        projector,
        projection, 
        scaler,
        verbose=False,
        **kwargs,
    ):
        super().__init__()
        self.verbose = verbose
        self.lr = kwargs.get("lr", 0.001)
        self.model = LatentDiffusionModel(
            latent_dim=kwargs.get("latent_dim", 8),
            hidden_dim=kwargs.get("hidden_dim", 64),
            nhidden=kwargs.get("nhidden", 5),
            timesteps=kwargs.get("timesteps", 100),
            time_embedding_dim=kwargs.get("time_embedding_dim", 8),
            epsilon=kwargs.get("epsilon", 0.008),
            target_embedding_dim=kwargs.get("target_embedding_dim", 8),
            dp_rate=kwargs.get("dp_rate", 0.1),
            verbose=verbose
        )
        self.noise_multiplier = kwargs.get("noise_multiplier", .1)
        # self.save_hyperparameters()
        self.decode = decode
        
        self.projector = projector
        self.projection = projection
        self.scaler = scaler
        self.timesteps = kwargs.get("timesteps", 100)

    def forward(self, x, y):
        if self.verbose:
            print('x', x.shape)
            print('y', y.shape)
        noise = torch.randn_like(x) * self.noise_multiplier
        pred_noise, noise_added = self.model(x, y, noise)
        return pred_noise, noise_added
    
    def _reverse_diffusion(self, x_t, y, t):
        noise = torch.randn_like(x_t)
        return self.model._reverse_diffusion(x_t, y, t, noise)

    def training_step(self, batch, batch_idx):
        x, y, num = batch
        pred_noise, noise = self.forward(x, y)
        # print('pred_noise', pred_noise.shape)
        loss = nn.functional.mse_loss(pred_noise, noise)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # torch.clip_grad_norm_(self.model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .1)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, num = batch
        pred_noise, noise = self.forward(x, y)
        loss = nn.functional.mse_loss(pred_noise, noise)


        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        if batch_idx == 0 and self.decode is not None:
            rand_idx = torch.randint(0, x.shape[0], (1,))
            rand_y = y[rand_idx]
            x_t, history = self.model.sampling(rand_y, clipped_reverse_diffusion=False, device='mps', noise_mul=self.noise_multiplier)
            print('history', history.shape)
            with torch.no_grad():
                # make image by decoding latent space
                x, y, file_num = batch
                if file_num is not None:
                    file_num_formatted = str(file_num[rand_idx].item())
                    file_num_formatted = '0'* (6 - len(file_num_formatted)) + file_num_formatted
                    path_text = '../stranger_repos/HumanML3D/HumanML3D/texts'
                    with open(f"{path_text}/{file_num_formatted}.txt", 'r') as f:
                        text = f.read().split('\n')[0].split('#')[0]

                t = torch.tensor([self.timesteps-1]*x.shape[0]).to('mps')
                noise = torch.randn_like(x) * self.noise_multiplier
                if self.verbose:
                    print('x', x.shape)
                    print('t', t.shape)
                    print('noise', noise.shape)
                

                x_t = torch.tensor(self.scaler.inverse_transform(x_t.cpu().detach().numpy())).to('mps')
                sample = self.decode(x_t)

                x_t_train = self.model._forward_diffusion(x, t, noise)
                sample_pure_noise = self.decode(torch.tensor(self.scaler.inverse_transform(x_t_train.cpu().detach().numpy())).to('mps'))


                path_base = self.logger.log_dir + f"/animations/recon_{self.current_epoch}"


                for data, name in zip([sample, sample_pure_noise], ['sample', 'sample_pure_noise', ]):
                    print(name, data.shape)
                    rand_idx = torch.randint(0, data.shape[0], (1,))
                    data_selected = data[rand_idx].cpu().detach().numpy().squeeze()
                    #print(name, data_selected.shape)
                    plot_3d_motion_animation(
                                data = data_selected,
                                title = text,
                                figsize=(10, 10),
                                fps=20,
                                radius=2,
                                save_path=f"{path_base}_{name}.mp4",
                                velocity=False
                            )
                    plt.close()

            # history plot
            history = history[0] # (n_samples=1, timesteps+1, latent_dim)
            # unscale
            history = self.scaler.inverse_transform(history.cpu().detach().numpy())
            #project
            history = self.projector.transform(history)
            # plot
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.scatter(self.projection[:, 0], self.projection[:, 1], c='black', s=10, label='projection')
            ax.plot(history[:, 0], history[:, 1], c='red', linewidth=2, label='history')
            fig.suptitle('Reversed diffusion-path in the latent space')
            plt.legend()
            ax.set_xlabel('UMAP component 1')
            ax.set_ylabel('UMAP component 2')
            plt.tight_layout()
            self.logger.experiment.add_figure('reversed_diffusion_path', fig, global_step=self.global_step)
            
            plt.close()



        return loss

    def test_step(self, batch, batch_idx):
        x, y, num = batch
        pred_noise, noise = self.forward(x, y)
        loss = nn.functional.mse_loss(pred_noise, noise) / self.noise_multiplier
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5, verbose=True)
        return [opt], [sch]
