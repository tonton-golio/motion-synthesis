import streamlit as st
import os, glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os

# Intro and title
"""
# Motion Autoencoder


"""
tab_names = [
    'VAE1',
]
tabs = {name: tab for name, tab in zip(tab_names, st.tabs(tab_names))}


with tabs['VAE1']:
    st.write("""
    Our first model, VAE1, is a transformer based VAE which uses linear layers for compression and decompression.
             """)
    



def find_saved_latent(path = f"logs/VAE/train/", cfg_name='hparams'):    
    """
    Find saved latent vectors from VAE training
    """

    VAE_data = {}
    # st.write(os.listdir(path))
    for version in os.listdir(path):
        if not os.path.isdir(f"{path}{version}"):
            continue
        version_num = version.split('_')[-1]
        contents = os.listdir(f"{path}{version}")
        base_path = os.path.join(path, version, )
        # st.write(contents)
        if 'saved_latent' in contents:
            # st.write(f"Found saved latent vectors for version {version_num}")
            cfg_file = None  # get config file
            for file in contents:
                if cfg_name in file and file.endswith('.yaml'):
                    cfg_file = file
                    break
            
            projection = None  # get projection image
            for file in contents:
                if 'projection' in file and file.endswith('.png'):
                    projection = file
                    break
            if projection is None:
                continue

            checkpoints = glob.glob(f"{base_path}/checkpoints/*")  # check for checkpoints
            saved_latent = os.listdir(os.path.join(base_path, 'saved_latent'))  # open saved_latent and check whats inside

            VAE_data[version_num] = {
                'saved_latent' : saved_latent,
                'paths' : {
                    'config' : os.path.join(base_path, cfg_file),
                    'saved_latent' : os.path.join(base_path, 'saved_latent'),
                    'projection' : os.path.join(base_path, projection) if projection else None,
                    'checkpoints' : checkpoints,
                    'log' : base_path,
                },
                'contents' : contents
            }

    return VAE_data

def load_log_scalar(logdir, tag='train/loss'):
    """
    Load the loss from tensorboard log
    """
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    tags = ea.Tags()['scalars']
    if tag not in tags:
        'try one of these:', tags
        raise ValueError(f"Tag {tag} not found in {logdir}")
    
    return ea.Scalars(tag)


def view_entry(entry):

    cols = st.columns([1.3, 1])

    sub_cols = cols[0].columns([1, 1])
    with sub_cols[0]:
        # get num param from num_params.txt
        with open(f"{VAE_data[str(idx)]['paths']['saved_latent']}/num_params.txt", 'r') as file:
            num_params = file.read()

        num_params

        # log contents
        with st.expander('Log contents'):
            st.write(VAE_data[str(idx)])

        # config
        from motion_latent_diffusion.utils import load_config
        cfg = load_config(VAE_data[str(idx)]['paths']['config'])
        with st.expander('Config'):
            cfg

    with sub_cols[1]:

        VAE_data[entry]['image'] = plt.imread(VAE_data[entry]['paths']['projection'])
        st.image(VAE_data[entry]['image'])


    with cols[0]:
        # load the loss

        # # "VELOCITY_L2_trn"
        # 1:"MOTION_L2_trn"
        # 2:"MOTIONRELATIVE_L2_trn"
        # 3:"DIVERGENCE_KL_trn"
        scalars = {
            k : load_log_scalar(VAE_data[str(idx)]['paths']['log'], tag=k) for k in ['MOTION_L2_trn', 'MOTIONRELATIVE_L2_trn', 'DIVERGENCE_KL_trn']
        }
        # load_log_scalar(VAE_data[str(idx)]['paths']['log'], tag='asd')
        
        
        # plot all scalars
        fig, axes = plt.subplots(1, len(scalars), figsize=(5*len(scalars), 5))
        for ax, (k, v) in zip(axes, scalars.items()):
            ax.plot([x.step for x in v], [x.value for x in v])
            ax.set_title(k)

        st.pyplot(fig)
    

    

    with cols[1]:
        # view recon_latest.mp4

        st.video(f"{VAE_data[str(idx)]['paths']['log']}/recon_latest.mp4")



VAE_data = find_saved_latent('motion_latent_diffusion/logs/MotionVAE/VAE1/train/')
idx = st.selectbox('Select VAE model', list(VAE_data.keys()))



# now we want to show the loss,
# the projection of the latent space


# show the projection

view_entry(idx)

st.divider()
'## Inference (sampling from the latent space and decoding)'
from motion_latent_diffusion.modules import MotionVAE
# load model
path = VAE_data[str(idx)]['paths']['checkpoints'][0]
model = MotionVAE.load_from_checkpoint(path)

