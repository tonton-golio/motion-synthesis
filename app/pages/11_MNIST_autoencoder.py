
import streamlit as st

# Intro and title
"""
# MNIST VAE
"""

tab_names = [
	'Results summary',
    'Latent Space',
	#'VAE weights',
]

tabs = {name: tab for name, tab in zip(tab_names, st.tabs(tab_names))}

with tabs['Results summary']:
	"""
	We have trained A conv-based VAE on the MNIST dataset, using a range of KL divergence weights. We obtain latent spaces of varying size. The larger latent spaces admit island formation, whereas a smaller latent space sees continent, of pangea formation.
	"""

	

with tabs['Latent Space']:
	# title and introduction
	"""
	### MNIST Latent Space

	show the following:
	* mnist latent space embedded in 2D
	* reconstruction from differnt part of latent space
	* reconstruction from traversal of latent space
	* metrics

	"""

	# # try loading it back
	# import torch
	import matplotlib.pyplot as plt

	# path = 'logs/MNIST_VAE/version_105/saved_latent/'
	# device = torch.device('mps')
	# z = torch.load(path + 'z.pt').to(device)
	# autoencoder = torch.load(path + 'model.pth').to(device)
	# # x = torch.load(path + 'x.pt').to(device)
	# y = torch.load(path + 'y.pt').to(device)
	# projector = torch.load(path + 'projector.pt')
	# projection = torch.load(path + 'projection.pt')

	# z.shape, y.shape
	# # plot the latent space
	# fig = plt.figure()
	# plt.scatter(projection[:, 0], #.detach().cpu().numpy(),
	# 			projection[:, 1], #.detach().cpu().numpy(),
	# 			c=y.detach().cpu().numpy(),
	# 			cmap='tab10',
	# 			alpha=0.5)

	# plt.colorbar()
	# st.pyplot(fig)

	from mnist_latent_diffusion.utils import find_saved_latent
	VAE_data = find_saved_latent(path = f"mnist_latent_diffusion/logs/VAE/train/")
	
	for entry in VAE_data:
		VAE_data[entry]['image'] = plt.imread(VAE_data[entry]['paths']['projection'])

		st.image(VAE_data[entry]['paths']['projection'])



# with tabs['VAE weights']:
# 	import streamlit as st
# 	import sys
# 	sys.path.append('..')
# 	from mnist_latent_diffusion.utils import get_ckpt, load_config
# 	import torch
# 	from mnist_latent_diffusion.modules.dataModules import MNISTDataModule


# 	# What we need to do?
# 	## Load a model (select it in the sidebar)
# 	## get sample inputs (scaled in the correct way)
# 	## Choose an input
# 	## View graph of network, with given input


# 	# Title and introductionc
# 	"""
# 	# MNIST VAE Weights

# 	This page allows you to load a pre-trained VAE model and view the weights of the model.
# 	"""

# 	# checkpoint paths

# 	with st.sidebar:
		
# 		checkpoints = get_ckpt(parent_log_dir='../mnist_latent_diffusion/logs/VAE/train/', return_all=True)
# 		checkpoint = st.selectbox('Select a checkpoint', list(checkpoints.keys()))

# 	# Load model
# 	model = torch.load(checkpoints[checkpoint]['ckpt_path'])
# 	'model loaded'
# 	config = load_config(checkpoints[checkpoint]['config_path'])
# 	'config loaded'
# 	# Get sample inputs
# 	dm = MNISTDataModule(**config['TRAIN']['DATA'], verbose=False)
# 	dm.setup()
# 	'mnist data module setup'

# 	batch = next(iter(dm.train_dataloader()))

# 	batch


# 	model