import streamlit as st
from transformers import CLIPProcessor, CLIPModel
import torch
"""
# Motion Latent Diffusion Inference

* first we need a text input for the prompt
* then we translate with clip
* the we load the LD scaler
* then we load the decoder
* then we load the LD model
"""

text = st.text_area("Enter a prompt", "A person walking")

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

tokens = processor(text, return_tensors="pt", padding=True, truncation=True)

tokens

clip = model.get_text_features(
       **tokens
    )

clip.shape

ckpt_path_LD = 'motion_latent_diffusion/logs/MotionLD/VAE1/version_67/checkpoints/epoch=999-step=750000.ckpt'

from motion_latent_diffusion.modules.LatentMotionData import LatentMotionData
from motion_latent_diffusion.modules.MotionLatentDiffusion import MotionLatentDiffusion

V = 77

# decoder = torch.load(f'../motion_latent_diffusion/logs/MotionLD/VAE1/version_{V}/decoder.pt')
# scaler = torch.load(f'../motion_latent_diffusion/logs/MotionLD/VAE1/version_{V}/scaler.pt')



# model = MotionLatentDiffusion(
#         decode=decoder,
#         scaler=scaler,
#         latent_dim=1024
#         **cfg["MODEL"]
#     )

model_path = f'motion_latent_diffusion/logs/MotionLD/VAE1/version_{V}/model.pt'
model = torch.load(model_path)

model.eval()
x_t = model.model.sampling(clip, clipped_reverse_diffusion=False, device='cpu', 
                           noise_mul=1.0
                           )

x_t.shape

sample = model.decode(torch.tensor(model.scaler.inverse_transform(x_t.cpu().detach().numpy())).to('mps')).squeeze(0)
sample = sample.detach().cpu().numpy()

# save sample as npy
import numpy as np
save_path_base = 'app/assets_produced/30_Motion_Latent_Diffusion_Inference/' 
fname = save_path_base+'_'.join(text.split())
np.save(fname+'.npy', sample)



import matplotlib.pyplot as plt
from motion_latent_diffusion.utils import plot_3d_motion_animation


plot_3d_motion_animation(
        data = sample,
        title = text,
        figsize=(10, 10),
        fps=20,
        radius=2,
        save_path=fname+'.mp4',
        velocity=False
                            )
plt.close()

st.video(save_path_base+'_'.join(text.split())+'.mp4')

# os.chdir('../app/')