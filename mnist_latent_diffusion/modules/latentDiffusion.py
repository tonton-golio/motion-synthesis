import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
# PCA
from sklearn.decomposition import PCA

import torchvision

class TimeMLP(nn.Module):
    '''
    naive introduce timestep information to feature maps with mlp
    '''
    def __init__(self,in_dim,hidden_dim,out_dim):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(in_dim,hidden_dim),
                                nn.SiLU(),
                                nn.Linear(hidden_dim,hidden_dim),
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
    naive introduce timestep information to feature maps with mlp
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
        verbose=False,
    ):
        super(SimpleModel, self).__init__()
        self.verbose = verbose
        time_ed = self.time_embedding_dim = time_embedding_dim
        targ_ed = self.target_embedding_dim = 10
        self.time_embedding = nn.Embedding(timesteps, time_embedding_dim)


        self.time_mlp=TimeMLP(
            in_dim=time_ed, 
            hidden_dim=time_ed*2,
            out_dim=time_ed)
        
        self.target_mlp=TargetMLP(
            embedding_dim=targ_ed,
            hidden_dim=targ_ed*2,
            out_dim=targ_ed,
            nhidden=3
        )

        dropout = nn.Dropout(dp_rate)

        in_dim = latent_dim + time_ed + targ_ed
        print('in_dim', in_dim, 'hidden_dim', hidden_dim, 'latent_dim', latent_dim, 'time_ed', time_ed, 'targ_ed', targ_ed)
        self.fc_noise = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            dropout,
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            dropout,
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            dropout,
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            dropout,
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        

    def forward(self, x, y, t):


        if self.verbose:
            print('x', x.shape)
            print('y', y.shape)
            print('t', t.shape)

        t = self.time_embedding(t)  # embed time
        # print('time embedding success')
        t = self.time_mlp(t)  # dim is hidden_dim
        # print('time mlp success')
        y = self.target_mlp(y)  # dim is hidden_dim
        # print('target mlp success')

        if self.verbose:
            print('t', t.shape)
            print('y', y.shape)

        cat = torch.cat([x, y, t], dim=-1)

        if self.verbose:
            print('cat', cat.shape)
                
        return self.fc_noise(cat)
    
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
        verbose=False,
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
            verbose=verbose,
        )

        self.target_embedding = nn.Embedding(10, 10)

    def forward(self, x, y, noise):
        # x:NCHW
        t = torch.randint(0, self.timesteps, (x.shape[0],)).to(x.device)

        x_t = self._forward_diffusion(x, t, noise)
        pred_noise = self.model(x_t, y, t)
        return pred_noise, noise, x_t, t

    @torch.no_grad()
    def sampling(self,n_samples, device="mps", tqdm_disable=True):
        
        x_t = torch.randn( (n_samples, self.latent_dim), device=device, dtype=torch.float32)
        hist = [x_t]

        y = torch.randint(0, 10, (n_samples,), device=device, dtype=torch.long)
        y_emb = self.target_embedding(y)

        for ti in tqdm(reversed(range(0, self.timesteps)),desc="Sampling", disable=tqdm_disable):
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
class LatentDiffusionModule(pl.LightningModule):
    def __init__(
        self,
        autoencoder=None,
        scaler=None,
        criteria=None,
        classifier=None,
        projectors={},
        projection=None,
        labels=None,
        verbose=False,
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
            verbose=verbose,
        )
        # self.device = torch.device("mps")
        self.noise_multiplier = kwargs.get("NOISE", 1.5)
        self.model.noise_multiplier = self.noise_multiplier
        if autoencoder is not None:
            autoencoder.model.eval()
            self.decoder = autoencoder.model.decode if autoencoder is not None else None

        self.scaler = scaler

        self.criteria = criteria
        self.classifier = classifier
        self.projector = projectors
        self.projection = projection
        self.labels = labels[:5000]

        # self.recon_loss = True if 'RECON_L2' in self.criteria.loss_weights.keys() else False
        
        self.use_label_for_decoder = kwargs.get("USE_LABEL_FOR_DECODER", False)
    
    def forward(self, data):
        x, y = data
        # print('forward: x', x.shape)
        # print('forward: y', y.shape)
        noise = torch.randn_like(x) * self.noise_multiplier
        return self.model(x, y, noise)

    def training_step(self, batch, batch_idx):
        # print('starting training step')
        res = self._common_step(batch, stage="train")
        # clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), .01)
        # print('done training step')
        return res["loss"]

    def _common_step(self, batch, stage='train'):
        x, y = batch
        pred_noise, noise, x_t, t = self.forward(batch)
        
        
        # if self.recon_loss:
        #     x_hat = x + noise - pred_noise

        #     # send through decoder
        #     if self.scaler is not None:
        #         x_hat = self.apply_scaler(x_hat, inverse=True, return_tensor_type=True)
        #         x = self.apply_scaler(x, inverse=True, return_tensor_type=True)
        #     with torch.no_grad():
        #         recon = self.decoder(x_hat)
        #         recon_gt = self.decoder(x)

        #     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        #     ax[0].imshow(recon[0].squeeze().detach().cpu().numpy(), cmap='gray')
        #     ax[0].set_title('recon')
        #     ax[1].imshow(recon_gt[0].squeeze().detach().cpu().numpy(), cmap='gray')
        #     ax[1].set_title('recon_gt')
        #     self.logger.experiment.add_figure(f'{stage}_recon', fig, global_step=self.global_step)

        # recon = self.decoder(torch.tensor(self.scaler.inverse_transform(x_hat.detach().cpu())).float().to("mps"))
        # recon_gt = self.decoder(torch.tensor(self.scaler.inverse_transform(x.detach().cpu())).float().to("mps"))
        # with torch.no_grad():
        
        # self.classifier.eval()
        # class_pred = self.classifier(x_hat)
        

        loss_data = {
            'NOISE_L2': {'rec': pred_noise, 'true': noise},
            #'CLASS_BCE': {'rec': class_pred, 'true': y}
        }
        # if self.recon_loss:
        #     loss_data['RECON_L2'] = {'rec': recon, 'true': recon_gt}


        loss, lss_scaled, lss_unscaled = self.criteria(loss_data)

        self.log_dict(lss_unscaled, prog_bar=True)
        self.log('total_loss', loss, prog_bar=True) 
        return dict(
            loss=loss,
            losses_scaled=lss_scaled,
            losses_unscaled=lss_unscaled,
            x=x,
            # x_hat=x_hat,
            y=y,
            # recon=recon,
            # recon_gt=recon_gt,
        )

    def validation_step(self, batch, batch_idx):
        # print('starting validation step')
        res = self._common_step(batch, stage="val")
        # print('Finish common step part of validation step')
        if batch_idx == 0 and self.current_epoch == 0:
            self.check_noise_level(batch)
        # print('done validation step')
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

    def apply_projector(self, x, projector, inverse=False, return_tensor_type=False):
        org_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if inverse:
            scaled = projector.inverse_transform(x.detach().cpu().numpy())
        else:
            scaled = projector.transform(x.detach().cpu().numpy())
        if return_tensor_type: 
            return torch.tensor(scaled).view(*org_shape[:-1], scaled.shape[-1]).to('mps')
        else:
            return scaled.reshape(*org_shape[:-1], scaled.shape[-1])

    def on_validation_epoch_end(self):
        # print('starting validation epoch end')
        with torch.no_grad():
            # Simplify y value assignment            
            # Sample from model
            # print('starting sampling')
            n_samples = 6
            n_time_steps_show = 4
            sample, hist, y_flags = self.model.sampling(n_samples, device='mps', tqdm_disable=True)
            # print('done sampling')    

            # lets make the other image.
            # a plot of the latent space, through the projector. with the history projected and shown as a line

            # project the latent space
            # print('projecting, hist.shape', hist.shape)
            if self.projector is not None:
                fig_prj, ax_prj = plt.subplots(1, 1, figsize=(10, 10))
                hist = self.apply_scaler(hist, inverse=True, return_tensor_type=True)
                hist_prj = self.apply_projector(hist)
                # print('hist_prj', hist_prj.shape)
                # plot the latent space
                # make sure lengths are the same
                if len(self.projection) < len(self.labels):
                    self.labels = self.labels[:len(self.projection)]
                elif len(self.projection) > len(self.labels):
                    self.projection = self.projection[:len(self.labels)]


                scat = ax_prj.scatter(self.projection[:, 0], self.projection[:, 1], c=self.labels, s=10, alpha=0.5, cmap='tab10')
                plt.colorbar(scat)
                for i in range(hist_prj.shape[1]):
                    ax_prj.plot(hist_prj[:, i, 0], hist_prj[:,i, 1], c='r', alpha=1, ls='--')

                self.logger.experiment.add_figure(f'hist latent projected', fig_prj,  global_step=self.global_step)

                plt.close()

                # now with PCA
                pca = PCA(n_components=2)
                hist_pca = pca.fit_transform(hist.view(-1, hist.shape[-1]).detach().cpu().numpy())

                hist_pca = hist_pca.reshape(hist.shape[0], hist.shape[1], -1)
                y_flags = y_flags.detach().cpu().numpy()
                print('hist_pca', hist_pca.shape)
                print('y_flags', y_flags.shape)

                fig_pca, ax_pca = plt.subplots(1, 1, figsize=(12, 7))
                scat = ax_pca.scatter(hist_pca[:, :, 0].flatten(), hist_pca[:, :, 1].flatten(),
                                      #c=y_flags,
                                      s=10, alpha=0.5, cmap='tab10')
                for i in range(hist_pca.shape[0]):
                    # if i == 0:
                    #     alpha = 1
                    #     lw = 2
                    #     label = 'first trajectory'
                    # else:
                    #     alpha = 0.5
                    #     label = None
                    #     lw = .5
                    plot_kwargs = {
                        'main': {'alpha': 1, 'lw': 2, 'label': 'main'},
                        'other': {'alpha': 0.5, 'lw': 1, 'label': None, 'ls': '--'}

                    }['main' if i == 0 else 'other']
                    if i > 2:continue
                    ax_pca.plot(hist_pca[i, :, 0], hist_pca[i, :, 1], **plot_kwargs)
                    

                    # plot points for start and finish
                    ax_pca.scatter(hist_pca[i, 0, 0], hist_pca[i, 1, 0], c='black', s=200, alpha=1)
                    ax_pca.scatter(hist_pca[i, 0, -1], hist_pca[i, 1, -1], c='r', s=2000, alpha=.4)
                    # ax_pca.text(hist_pca[i, 0], hist_pca[i, 1], f'{y_flags[i].item()}', fontsize=12)
                    break

                plt.colorbar(scat)
                plt.title('PCA of latent space')
                self.logger.experiment.add_figure(f'hist latent pca', fig_pca,  global_step=self.global_step)




            # Ensure hist is not empty and prepare it for plotting
            if len(hist) < n_time_steps_show:
                n_time_steps_show = len(hist)
            
            # print('hist', hist.shape)
            timesteps_hist = torch.arange(0, hist.shape[0]).to('mps')

            idx_show = torch.linspace(0, hist.shape[0]-1, steps=n_time_steps_show, dtype=torch.long)

            # print('idx_show', idx_show)
            hist = hist[idx_show]
            timesteps_hist = timesteps_hist[idx_show]
            # print('timesteps_hist', timesteps_hist)
            timesteps_hist = list(idx_show)[::-1]
            # print('timesteps_hist', timesteps_hist)
            
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
                    # if self.use_label_for_decoder:
                    #     hist_expanded.append(self.decoder(torch.tensor(hist[i]).to('mps'), y_flags))
                    # else:

                    hist_expanded.append(self.decoder(hist[i].to('mps')))

                hist = torch.stack(hist_expanded, dim=0).squeeze().cpu()

                # print('hist', hist.shape)


            # Create a figure with subplots
            if len(hist.shape) == 3:
                hist = hist.unsqueeze(0)
            # print('hist', hist.shape)
            rows, cols = hist.shape[:2]
            fig, axes = plt.subplots(rows, cols, figsize=(10, 10 * rows / cols))
            if rows == 1:
                axes = axes.reshape(1, axes.shape[0])
            # for i in range(rows):
            #     for j in range(cols):
            #         ax[i, j].imshow(hist[i, j], cmap='gray')
            #         ax[i, j].axis('off')
            for i in range(rows):
                for j in range(cols):
                    axes[i, j].imshow(hist[i, j].detach().cpu().numpy(), cmap='gray')
                    axes[i, j].set_xticks([])
                    axes[i, j].set_yticks([])

                    axes[i, j].set_title(f'y={y_flags[j].item()}')
                axes[i, 0].set_ylabel(f't={timesteps_hist[i].item()}')
                    
            
            # # Set top row titles to y_flags
            # if y_flags is not None and len(y_flags) == cols:
            #     for i, flag in enumerate(y_flags):
                    # axes[0, i].set_title(f'y={flag.item()}')
                    
            self.logger.experiment.add_figure(f'hist latent', fig,  global_step=self.global_step)
            plt.close()

        # print('done validation epoch end')


    def test_step(self, batch, batch_idx):
        pred_noise, noise, x_t, t = self.forward(batch)
        loss = nn.functional.mse_loss(pred_noise, noise) / self.noise_multiplier
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        # return torch.optim.AdamW(self.parameters(), lr=self.lr)
        # decrease lr by 0.1 every 10 epochs
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5, verbose=True)
        return [optimizer], [scheduler]


    @torch.no_grad()
    def check_noise_level(self, batch, N=3, nt=10):

        x, y = batch
        print('x', x.shape)
        print('y', y.shape)

        # select N samples
        x = x[:N]
        y = y[:N]

        # get y label, from one hot to index
        y = y.argmax(1)
        print('x', x.shape)
        print('y', y.shape, y)

        x_decoded = self.decoder(x)

        # make t an interger linspace
        if self.model.timesteps < nt:
            nt = self.model.timesteps

        x = x.view(N, 1, *x.shape[1:]).repeat(1, nt, 1)
        print('x', x.shape)
        t = torch.linspace(0, self.model.timesteps, steps=nt, dtype=torch.long).to('mps')
        t = t.repeat(N, 1)
        print('t', t.shape)

       
        
        # flatten all three
        x = x.view(N*nt, *x.shape[2:])
        t = t.view(N*nt)

        # add noise
        noise = torch.randn_like(x) * self.noise_multiplier
        x_t = self.model._forward_diffusion(x, t, noise)

        
        # decode
        x_t = self.decoder(x_t)
        # reshape
        x_t = x_t.view(N, nt, *x_t.shape[1:])

        #replace x_t with x_decoded for t=0
        x_t[:, 0] = x_decoded

        # grid = torchvision.utils.make_grid(x_t, nrow=N, cmap='gray')
        fig, ax = plt.subplots(N, nt, figsize=(10,3))
        for i in range(N):
            for j in range(nt):
                ax[i, j].imshow(x_t[i, j].squeeze().detach().cpu().numpy(), cmap='gray_r')
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                if i == 0:
                    ax[i, j].set_title(f't={t[j]}')
                if j == 0:
                    ax[i, j].set_ylabel(f'y={y[i]}', rotation=0, labelpad=20)
        plt.tight_layout()
        self.logger.experiment.add_figure("x_t", fig, global_step=self.global_step)

