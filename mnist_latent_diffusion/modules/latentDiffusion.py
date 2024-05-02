import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

import torchvision

class TimeMLP(nn.Module):
    '''
    naive introduce timestep information to feature maps with mlp and add shortcut
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
        layers.append(act)


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


        self.time_mlp=TimeMLP(time_embedding_dim, time_embedding_dim*2,time_embedding_dim)
        self.target_mlp=TargetMLP(embedding_dim=10,hidden_dim=hidden_dim,out_dim=latent_dim*4, nhidden=nhidden)


        self.fc_noise = nn.Sequential(
            nn.Linear(4*latent_dim + latent_dim + time_embedding_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        

    def forward(self, x, y, t):
        # print('x', x.shape)
        # print('SimpleModel: y', y.shape)
        # print('SimpleModel: t', t.shape)
        # print('SimpleModel: x', x.shape)
        t = self.time_embedding(t)  # embed time
        # print('time embedding success')
        t = self.time_mlp(t)  # dim is hidden_dim
        # print('time mlp success')
        y = self.target_mlp(y)  # dim is hidden_dim
        # print('target mlp success')

        # print('x', x.shape)
        # print('y', y.shape)

        cat = torch.cat([x, y, t], dim=-1)
        # print('x', x.shape)
        pred_x = self.fc_noise(cat)
        pred_noise = pred_x - x
        # print('fc_noise success, x', x.shape)
        return pred_noise
    
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

        # betas = self._cosine_variance_schedule(timesteps, epsilon)
        betas = self._linear_variance_schedule(timesteps, beta_start=1e-4, beta_end=0.02)
        # print('betas', betas.shape)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=-1)

        # print('alphas_cumprod', alphas_cumprod.shape)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        self.model = SimpleModel(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            nhidden=nhidden,
            timesteps=timesteps,
            time_embedding_dim=time_embedding_dim,
            # target_embedding_dim=target_embedding_dim,
        )

        self.target_embedding = nn.Embedding(10, 10)

    def forward(self, x, y, noise):
        # x:NCHW
        t = torch.randint(10, self.timesteps, (x.shape[0],)).to(x.device)

        x_t = self._forward_diffusion(x, t, noise)
        
        # print('x_t?', x_t.shape)
        pred_noise = self.model(x_t, y, t)
        noise_added = x_t - x
        return pred_noise, noise, x_t, t

    @torch.no_grad()
    @torch.no_grad()
    def sampling(self,n_samples, device="mps", tqdm_disable=True):
        
        x_t = torch.randn( (n_samples, self.latent_dim), device=device, dtype=torch.float32)
        hist = [x_t]

        y = torch.randint(0, 10, (n_samples,), device=device, dtype=torch.long)
        y_emb = self.target_embedding(y)

        for ti in tqdm(reversed(range(1, self.timesteps)),desc="Sampling", disable=tqdm_disable):
            noise=torch.randn_like(x_t).to(device) * .3
            t = torch.ones(n_samples, device=device, dtype=torch.long) * ti
            x_t=self._reverse_diffusion(x_t, y_emb, t, noise)
            hist.append(x_t)

        return x_t, torch.stack(hist,dim=0), y

    def _cosine_variance_schedule(self, timesteps, epsilon=0.008):
        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        f_t = (
            torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5)
            ** 2
        )
        betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)

        return betas
    
    def _linear_variance_schedule(self, timesteps, beta_start=1e-4, beta_end=0.02):
        betas = torch.linspace(beta_start, beta_end, steps=timesteps + 1, dtype=torch.float32) + .02
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
        pred_noise = self.model(x_t, y, t)
        alpha_t = self.alphas[t][:, None]
        beta_t = self.betas[t][:, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        x_t_copy = x_t.clone()
        x_t = 1 / torch.sqrt(alpha_t) * (x_t - ((1-alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred_noise) + torch.sqrt(beta_t) * noise

        # replace x_t with origional if t=0
        x_t = torch.where(t[:, None] == 0, x_t_copy, x_t)

        return x_t


# make pl model
class LatentDiffusionModel(pl.LightningModule):
    def __init__(
        self,
        autoencoder=None,
        scaler=None,
        criteria=None,
        classifier=None,
        projector=None,
        projection=None,
        labels=None,
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
        self.noise_multiplier = kwargs.get("NOISE", 1.0)
        self.model.noise_multiplier = self.noise_multiplier
        self.decoder = autoencoder.model.decode if autoencoder is not None else None
        self.scaler = scaler

        self.criteria = criteria
        self.classifier = classifier
        self.projector = projector
        self.projection = projection
        self.labels = labels

        self.recon_loss = True if 'RECON_L2' in self.criteria.loss_weights.keys() else False
        
        self.use_label_for_decoder = kwargs.get("USE_LABEL_FOR_DECODER", False)
    
    def forward(self, data):
        x, y = data
        # print('forward: x', x.shape)
        # print('forward: y', y.shape)
        noise = torch.randn_like(x) * self.noise_multiplier
        return self.model(x, y, noise)

    def training_step(self, batch, batch_idx):
        res = self._common_step(batch, stage="train")
        # clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), .01)
        return res["loss"]

    def _common_step(self, batch, stage='train'):
        pred_noise, noise, x_t, t = self.forward(batch)
        x, y = batch
        
    
        if self.recon_loss:
            x_hat = x + noise - pred_noise

            # send through decoder
            if self.scaler is not None:
                x_hat = self.apply_scaler(x_hat, inverse=True, return_tensor_type=True)
                x = self.apply_scaler(x, inverse=True, return_tensor_type=True)
            with torch.no_grad():
                recon = self.decoder(x_hat)
                recon_gt = self.decoder(x)

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(recon[0].squeeze().detach().cpu().numpy(), cmap='gray')
            ax[0].set_title('recon')
            ax[1].imshow(recon_gt[0].squeeze().detach().cpu().numpy(), cmap='gray')
            ax[1].set_title('recon_gt')
            self.logger.experiment.add_figure(f'{stage}_recon', fig, global_step=self.global_step)

        # recon = self.decoder(torch.tensor(self.scaler.inverse_transform(x_hat.detach().cpu())).float().to("mps"))
        # recon_gt = self.decoder(torch.tensor(self.scaler.inverse_transform(x.detach().cpu())).float().to("mps"))
        # with torch.no_grad():
        
        # self.classifier.eval()
        # class_pred = self.classifier(x_hat)
        

        loss_data = {
            'NOISE_L2': {'rec': pred_noise, 'true': noise},
            #'CLASS_BCE': {'rec': class_pred, 'true': y}
        }
        if self.recon_loss:
            loss_data['RECON_L2'] = {'rec': recon, 'true': recon_gt}


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
        
        if batch_idx == 0 and self.current_epoch == 0:
            self.check_noise_level(batch)

        return res["loss"]

    def apply_scaler(self, x, inverse=False, return_tensor_type=False):
        if self.scaler is None:
            return x
        org_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if inverse:
            scaled = self.scaler.inverse_transform(x.detach().cpu().numpy())
        else:
            scaled = self.scaler.transform(x.detach().cpu().numpy())

        if return_tensor_type: 
            return torch.tensor(scaled).view(org_shape).to('mps')
        else:
            return scaled

    def apply_projector(self, x, inverse=False, return_tensor_type=False):
        org_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if inverse:
            scaled = self.projector.inverse_transform(x.detach().cpu().numpy())
        else:
            scaled = self.projector.transform(x.detach().cpu().numpy())
        if return_tensor_type: 
            return torch.tensor(scaled).view(*org_shape[:-1], scaled.shape[-1]).to('mps')
        else:
            return scaled.reshape(*org_shape[:-1], scaled.shape[-1])

    def on_validation_epoch_end(self):
        with torch.no_grad():
            # Simplify y value assignment            
            # Sample from model
            print('starting sampling')
            n_samples = 6
            n_time_steps_show = 8
            sample, hist, y_flags = self.model.sampling(n_samples, device='mps', tqdm_disable=True)
            print('done sampling')

            # lets make the other image.
            # a plot of the latent space, through the projector. with the history projected and shown as a line

            # project the latent space
            print('projecting, hist.shape', hist.shape)
            if self.projector is not None:
                fig_prj, ax_prj = plt.subplots(1, 1, figsize=(10, 10))
                hist = self.apply_scaler(hist, inverse=True, return_tensor_type=True)
                hist_prj = self.apply_projector(hist)
                print('hist_prj', hist_prj.shape)
                # plot the latent space
                ax_prj.scatter(self.projection[:, 0], self.projection[:, 1], c=self.labels, s=2, alpha=0.5)
                for i in range(hist_prj.shape[1]):
                    ax_prj.plot(hist_prj[:, i, 0], hist_prj[:,i, 1], c='r', alpha=1, ls='--')

                self.logger.experiment.add_figure(f'hist latent projected', fig_prj,  global_step=self.global_step)

                plt.close()



            # Ensure hist is not empty and prepare it for plotting
            if len(hist) < n_time_steps_show:
                n_time_steps_show = len(hist)
            hist = hist[::len(hist) // n_time_steps_show]#.squeeze().cpu()
            
            # decode
            if self.decoder is not None:
                # reshape
                ## current shapes: hist: (timesteps, n_samples, latent_dim), y_flags: (n_samples)
                ## desired shapes: hist: (n_samples* timesteps, latent_dim), y_flags: (n_samples*timesteps)
                # y_flags_rep = y_flags.repeat(hist.shape[0])
                # hist = hist.view(-1, hist.shape[-1]).to('mps')

                # print('hist', hist.shape)
                hist_expanded = []
                # sample = self.decoder(sample)
                for i in range(len(hist)):
                    if self.use_label_for_decoder:
                        hist_expanded.append(self.decoder(torch.tensor(hist[i]).to('mps'), y_flags))
                    else:

                        hist_expanded.append(self.decoder(torch.tensor(hist[i]).to('mps')))

                hist = torch.stack(hist_expanded, dim=0).squeeze().cpu()

                # print('hist', hist.shape)


            # Create a figure with subplots
            rows, cols = hist.shape[:2]
            fig, axes = plt.subplots(rows, cols, figsize=(10, 10 * rows / cols))
            
            # for i in range(rows):
            #     for j in range(cols):
            #         ax[i, j].imshow(hist[i, j], cmap='gray')
            #         ax[i, j].axis('off')
            for i in range(rows):
                for j in range(cols):
                    axes[i, j].imshow(hist[i, j], cmap='gray')
                    axes[i, j].axis('off')
                    axes[i, j].set_title(f'y={y_flags[j].item()}')
                    
            
            # # Set top row titles to y_flags
            # if y_flags is not None and len(y_flags) == cols:
            #     for i, flag in enumerate(y_flags):
                    # axes[0, i].set_title(f'y={flag.item()}')
                    
            self.logger.experiment.add_figure(f'hist latent', fig,  global_step=self.global_step)
            plt.close()


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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, verbose=False)
        return [optimizer], [scheduler]


    @torch.no_grad()
    def check_noise_level(self, batch, N=8):
        x, y = batch
        x = x[:N]
        y = y[:N]

        x = x[2:3]
        x = x.repeat(N, 1)
        y = y[2:3].argmax(1).repeat(N)
        
        # print(x.shape)
        # print(y.shape)

        noise = torch.randn_like(x)
        # make t an interger linspace
        t = torch.linspace(0, self.model.timesteps, steps=N, dtype=torch.long).to('mps')
        x_t = self.model._forward_diffusion(x, t, noise)

        # decode
        if self.use_label_for_decoder:
            x_t = self.decoder(x_t, y)
        else:
            x_t = self.decoder(x_t)

        grid = torchvision.utils.make_grid(x_t, nrow=3)
        self.logger.experiment.add_image("x_t", grid, global_step=self.global_step)

