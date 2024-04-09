import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from tqdm import tqdm

import torchvision

class TimeMLP(nn.Module):
    '''
    naive introduce timestep information to feature maps with mlp and add shortcut
    '''
    def __init__(self,embedding_dim,hidden_dim,out_dim):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(embedding_dim,hidden_dim),
                                nn.SiLU(),
                               nn.Linear(hidden_dim,out_dim))
        self.act=nn.SiLU()

    def forward(self,x,t):
        t_emb=self.mlp(t)#.unsqueeze(-1).unsqueeze(-1)
        # print('t_emb', t_emb.shape)
        x=x+t_emb
  
        return self.act(x)
    
class TargetMLP(nn.Module):
    '''

    '''
    def __init__(self,embedding_dim,hidden_dim,out_dim, nhidden=5, act=nn.LeakyReLU() ):
        super().__init__()
        layers = [nn.Linear(embedding_dim,hidden_dim), 
                  act]
        for i in range(nhidden):
            layers.append(nn.Linear(hidden_dim,hidden_dim))
            layers.append(act)

        layers.append(nn.Linear(hidden_dim,out_dim))


        self.mlp=nn.Sequential(*layers)

    def forward(self,t):
        return self.mlp(t)


class SimpleModel(nn.Module):
    def __init__(
        self,
        latent_dim,
        hidden_dim,
        nhidden=5,
        timesteps=10,
        time_embedding_dim=64,
        dp_rate=0.1,
    ):
        super(SimpleModel, self).__init__()
        self.time_embedding_dim = time_embedding_dim
        self.target_embedding_dim = 10
        self.time_embedding = nn.Embedding(timesteps, time_embedding_dim)


        self.time_mlp=TimeMLP(time_embedding_dim,hidden_dim,latent_dim)
        self.target_mlp=TargetMLP(embedding_dim=10,hidden_dim=hidden_dim,out_dim=hidden_dim, nhidden=nhidden)


        self.fc_noise = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        

    def forward(self, x, y, t):
        # print('x', x.shape)
        t = self.time_embedding(t)  # embed time
        x = self.time_mlp(x, t)  # dim is hidden_dim

        y = self.target_mlp(y)  # dim is hidden_dim

        # print('x', x.shape)
        # print('y', y.shape)

        x = torch.cat([x, y], dim=-1)
        x = self.fc_noise(x)
        return x
    
class LatentDiffusion(nn.Module):
    def __init__(
        self,
        latent_dim=8,
        hidden_dim=64,
        nhidden=3,
        timesteps=1000,
        time_embedding_dim=64,
        target_embedding_dim=5,
        epsilon=0.008,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.latent_dim = latent_dim

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
            # target_embedding_dim=target_embedding_dim,
        )

    def forward(self, x, y, noise):
        # x:NCHW
        t = torch.randint(0, self.timesteps, (x.shape[0],)).to(x.device)
        x_t = self._forward_diffusion(x, t, noise)

        pred_noise = self.model(x_t, y, t)
        return pred_noise

    @torch.no_grad()
    def sampling(self, n_samples, y=None, device="mps"):
        x_t = torch.randn(
            (n_samples, self.latent_dim,)
        ).to(device)
        x_orig = x_t.clone()
        x_clean = x_t.clone()
        history = []
        if y is None:
            y = torch.randint(0, 10, (n_samples,)).to(device)
        else:
            y = torch.tensor(y).to(device).repeat(n_samples)

        y = torch.nn.functional.one_hot(y.long(), num_classes=10).float()
        print()
        orig_noised = False
        for i in tqdm(range(self.timesteps - 1, -1, -1), desc="Sampling", disable=True):
            noise = torch.randn_like(x_t).to(device) * self.noise_multiplier
            t = torch.tensor([i for _ in range(n_samples)]).to(device)
            if not orig_noised:
                x_orig = self._forward_diffusion(x_orig, 
                                                 t,
                                                   noise)
                orig_noised = True
                
            t = torch.tensor([i for _ in range(n_samples)]).to(device)
            x_t = self._reverse_diffusion(x_t, y, t, noise)

            history.append(x_t)

        history = torch.stack(history, dim=0)
        # x_t = (x_t + 1.0) / 2.0  # [-1,1] to [0,1]

        return x_t, history, y, x_orig, x_clean

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
        pred = self.model(x_t, y, t)

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

        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(
                x_t.shape[0], 1
            )  # ,1,1)
            std = torch.sqrt(
                beta_t * (1.0 - alpha_t_cumprod_prev) / (1.0 - alpha_t_cumprod)
            )
        else:
            std = 0.0

        return mean + std * noise


# make pl model
class LatentDiffusionModel(pl.LightningModule):
    def __init__(
        self,
        autoencoder=None,
        scaler=None,
        criteria=None,
        classifier=None,
        **kwargs,
    ):
        super().__init__()
        self.lr = kwargs.get("lr", 0.001)
        self.model = LatentDiffusion(
            latent_dim=kwargs.get("LATENT_DIM", 8),
            hidden_dim=kwargs.get("HIDDEN_DIM", 64),
            nhidden=kwargs.get("N_HIDDEN", 5),
            timesteps=kwargs.get("TIMESTEPS", 100),
            time_embedding_dim=kwargs.get("TIME_EMBEDDING", 8),
            epsilon=kwargs.get("EPSILON", 0.008),
            target_embedding_dim=kwargs.get("TARGET_EMBEDDING", 8),
        )
        # self.device = torch.device("mps")
        self.noise_multiplier = kwargs.get("NOISE", 3.0)
        self.model.noise_multiplier = self.noise_multiplier
        self.decoder = autoencoder.decode if autoencoder is not None else None
        self.scaler = scaler

        self.criteria = criteria
        self.classifier = classifier
    
    def forward(self, data):
        x, y = data
        noise = torch.randn_like(x) * self.noise_multiplier
        return self.model(x, y, noise), noise

    def training_step(self, batch, batch_idx):
        res = self._common_step(batch, stage="train")

        # clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
        return res["loss"]

    def _common_step(self, batch, stage='train'):
        pred_noise, noise = self.forward(batch)

        # recon = x - pred_noise
        x, y = batch
        x_hat = x - pred_noise + noise

        # recon = self.decoder(torch.tensor(self.scaler.inverse_transform(x_hat.detach().cpu())).float().to("mps"))
        # recon_gt = self.decoder(torch.tensor(self.scaler.inverse_transform(x.detach().cpu())).float().to("mps"))
        # with torch.no_grad():
        self.classifier.eval()
        class_pred = self.classifier(x_hat)
        

        loss_data = {
            'NOISE_L2': {'rec': pred_noise, 'true': noise},
            'CLASS_BCE': {'rec': class_pred, 'true': y}
        }

        loss, lss_scaled, lss_unscaled = self.criteria(loss_data)

        self.log_dict(lss_unscaled, prog_bar=True)
        self.log('total_loss', loss, prog_bar=True) 
        return dict(
            loss=loss,
            losses_scaled=lss_scaled,
            losses_unscaled=lss_unscaled,
            x=x,
            x_hat=x_hat,
            y=y,
            # recon=recon,
            # recon_gt=recon_gt,
        )

    def validation_step(self, batch, batch_idx):
        res = self._common_step(batch, stage="val")
        return res["loss"]

    def on_validation_epoch_end(self):
        with torch.no_grad():
            x_t, history, y, x_orig, x_clean = self.model.sampling(200, device="mps")
            # sample
            if self.scaler is not None:
                x_t = self.scaler.inverse_transform(x_t.cpu())
                x_t = torch.tensor(x_t).float().to("mps")

                x_orig = self.scaler.inverse_transform(x_orig.cpu())
                x_orig = torch.tensor(x_orig).float().to("mps")

                x_clean = self.scaler.inverse_transform(x_clean.cpu())
                x_clean = torch.tensor(x_clean).float().to("mps")
            

            recon_t = self.decoder(x_t)
            recon_orig = self.decoder(x_orig)
            recon_clean = self.decoder(x_clean)

            # accuracy in y vs class_pred
            class_pred = self.classifier(x_t)
            acc = (class_pred.argmax(dim=1) == y.argmax(dim=1)).float().mean()
            self.log("val_acc", acc, prog_bar=True)


            grid = torchvision.utils.make_grid(
                torch.cat([recon_clean[:8], recon_orig[:8], recon_t[:8]], dim=0),
                nrow=8,
                normalize=False,
            )

            self.logger.experiment.add_image(
                "clean, noisy, cleaned", grid, global_step=self.global_step
            )


    def test_step(self, batch, batch_idx):
        pred_noise, noise = self.forward(batch)
        loss = nn.functional.mse_loss(pred_noise, noise) / self.noise_multiplier
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        # return torch.optim.AdamW(self.parameters(), lr=self.lr)
        # decrease lr by 0.1 every 10 epochs
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.25, verbose=False)
        return [optimizer], [scheduler]

