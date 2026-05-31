import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import torch

import pytorch_lightning as pl
# logger
from pytorch_lightning.loggers import TensorBoardLogger


from motion_latent_diffusion.modules.LatentMotionData import LatentMotionData
from motion_latent_diffusion.modules.MotionLatentDiffusion import MotionLatentDiffusion
from motion_latent_diffusion.utils import test_translate, get_ckpt, plot_3d_motion_animation, load_config


def _device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

DEVICE = _device()


# get latent vectors
def find_saved_latent(path = f"motion_latent_diffusion/logs/MotionVAE/VAE1/train/", cfg_name='config'):
    """
    Find saved latent vectors from VAE training
    """

    VAE_data = {}
    for version in os.listdir(path):
        if not os.path.isdir(f"{path}{version}"):
            continue
        version_num = version.split('_')[-1]
        contents = os.listdir(f"{path}{version}")
        base_path = os.path.join(path, version, )
        # print(contents)
        if 'saved_latent' in contents:
            print(f"Found saved latent vectors for version {version_num}")
            cfg_file = None  # get config file
            for file in contents:
                if cfg_name in file and file.endswith('.yaml'):
                    cfg_file = file
                    break
            
            projection = None  # get projection image
            for file in contents:
                # print(file)
                if 'projection' in file and file.endswith('.png'):
                    projection = file
                    break

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

def show_saved_latent_info(data, return_fig=False):

    saved_latent_info = {}

    for version, info in data.items():
        saved_latent = info['saved_latent']
        saved_latent_info[version] = {
            'num_files' : len(saved_latent),
            'size' : None,
            'min' : None,
            'max' : None,
            'std_dev' : None,
            'projection' : None
        }

        for file in saved_latent:
            # get size of file
            # get min and max values
            # get std dev
            pass

        # projection_image = plt.imread(info['paths']['projection'])
        saved_latent_info[version]['projection'] = info['paths']['projection']

    fig, ax = plt.subplots(2, len(data), figsize=(20, 10))
    if len(data) == 1:
        ax = ax.reshape(2, 1)
    for i, (version, info) in enumerate(data.items()):
        if info['paths']['projection']: 
            ax[0, i].imshow(plt.imread(info['paths']['projection']))
            ax[0, i].set_title(f"Version {version}")
            ax[0, i].axis('off')

            ax[1, i].text(0.5, 0.5, f"Num Files: {saved_latent_info[version]['num_files']}", ha='center', va='center')
            ax[1, i].axis('off')
    if return_fig:
        return fig

    plt.show()

def latent_picker(path, cfg_name='config', show=True):
    data = find_saved_latent(path, cfg_name)
    # print(data)

    # if the user has difficulty picking a version, show info
    ## info to show: projection image, config file, saved_latent vectors (how many?, how big?, min/max values?, std dev?, etc.)
    ## also show the checkpoint files
    if len(data) == 0:
        print("No saved latent vectors found.")
        return None, None
    elif len(data) == 1:
        print("Only one version found.")
        version = list(data.keys())[0]
        return data[version], version
    
    else:
        if show:
            show_saved_latent_info(data)

        # ask user for input of version number
        print("Please enter the version number you would like to use: ")
        for version in data.keys():
            print(f"\t{version}")
        version = input('Version: ')

        return data[version], version

def load_latent(data_version):
    path = data_version['paths']['saved_latent']
    z_test = torch.load(path + '/latent_test.pt').to(DEVICE)
    z_train = torch.load(path + '/latent_train.pt').to(DEVICE)
    z_val = torch.load(path + '/latent_val.pt').to(DEVICE)
    
    text_test = torch.load(path + '/clip_test.pt')
    text_train = torch.load(path + '/clip_train.pt')
    text_val = torch.load(path + '/clip_val.pt')

    file_num_test = torch.load(path + '/file_nums_test.pt')
    file_num_train = torch.load(path + '/file_nums_train.pt')
    file_num_val = torch.load(path + '/file_nums_val.pt')
    
    autoencoder = torch.load(path + '/model.pth').to(DEVICE)
    projector = torch.load(path + '/projector.pt')
    projection = torch.load(path + '/projection.pt')

    # load checkpoint
    # checkpoint = torch.load(data_version['paths']['checkpoints'][0])
    # autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])

    return dict(
        z_train=z_train,
        z_val=z_val,
        z_test=z_test,
        texts_train=text_train,
        texts_val=text_val,
        texts_test=text_test,
        file_nums_train=file_num_train,
        file_nums_val=file_num_val,
        file_nums_test=file_num_test,
        autoencoder=autoencoder,
        projector=projector,
        projection=projection
    )

class LatentDecoder:
    def __init__(self, autoencoder, VAE_version, seq_len=160):
        self.autoencoder = autoencoder
        self.VAE_version = VAE_version
        self.seq_len = seq_len

    def decode(self, z):
        return decode_latent(z, self.autoencoder, self.VAE_version, self.seq_len)

    def __call__(self, z):
        return self.decode(z)

def decode_latent(z, autoencoder, VAE_version, seq_len=160):
    """Decode a batch of flat latents (B, latent_dim_flat) back to motion (B, T, 22, 3)."""
    autoencoder.eval()
    z = z.to(DEVICE)
    if z.dim() == 1:
        z = z.unsqueeze(0)
    bs = z.shape[0]
    lengths = torch.full((bs,), seq_len, device=DEVICE, dtype=torch.long)

    if VAE_version in ('VAEMLD', 'MLD'):
        # restore the (B, latent_size, latent_dim) shape the decoder expects
        ld = autoencoder.model.latent_dim
        z = z.reshape(bs, -1, ld)
        return autoencoder.decode(z, lengths)
    if VAE_version == 'VAE1':
        return autoencoder.decode(z)  # legacy, no lengths
    # legacy VAE4/VAE5: pass the true seq_len (NOT the old hardcoded 420/200)
    return autoencoder.model.decode(z, lengths)

# train
def train(VAE_version = 'VAE5'):

    # load config and instantiate logger
    cfg = load_config('motion_LD')
    logger = TensorBoardLogger('motion_latent_diffusion/logs', name=f'MotionLD/{VAE_version}')


    # make animations folder
    if not os.path.exists(f'motion_latent_diffusion/logs/MotionLD/{VAE_version}/animations'):
        os.makedirs(logger.log_dir + '/animations')
    # load latent vectors
    data_version, version = latent_picker(f'motion_latent_diffusion/logs/MotionVAE/{VAE_version}/train/', cfg_name='hparams', show=False)
    res_loaded = load_latent(data_version)

    z_train = res_loaded['z_train']
    z_val = res_loaded['z_val']
    z_test = res_loaded['z_test']
    texts_train = res_loaded['texts_train']
    texts_val = res_loaded['texts_val']
    texts_test = res_loaded['texts_test']

    file_num_train = res_loaded['file_nums_train']
    file_num_val = res_loaded['file_nums_val']
    file_num_test = res_loaded['file_nums_test']

    autoencoder = res_loaded['autoencoder']

    projector = res_loaded['projector']
    projection = res_loaded['projection']



    # data module
    data_module = LatentMotionData(z_train, z_val, z_test, 
                                   texts_train, texts_val, texts_test, 
                                   file_num_train, file_num_val, file_num_test,
                                   **cfg["DATA"]) 
                                  
    data_module.setup()

    scaler = data_module.scaler

    # save scaler
    torch.save(scaler, logger.log_dir + '/scaler.pt')
    
    # decoder
    seq_len = getattr(getattr(autoencoder, "model", autoencoder), "seq_len", 160)
    decoder = LatentDecoder(autoencoder, VAE_version, seq_len=seq_len)

    # save decoder
    torch.save(decoder, logger.log_dir + '/decoder.pt')

    # model
    model = MotionLatentDiffusion(
        decode=decoder,
        scaler=scaler,
        projection=projection,
        projector=projector,
        latent_dim=data_module.latent_dim,
        **cfg["MODEL"]
    )

    # train
    
    ckpt = None
    if cfg['FIT']['load_checkpoint']:
        path = logger.log_dir.split("version_")[0]
        ckpt = get_ckpt(path)

    trainer = pl.Trainer(**cfg["TRAINER"], logger=logger)
    trainer.fit(model, data_module, ckpt_path=ckpt)

    # test
    trainer.test(model, data_module)
    torch.save(model, logger.log_dir + '/model.pt')


@torch.no_grad()
def predict(clip_embedding, model, decoder, scaler=None, cfg_scale=2.5,
            save_path="recon_text.mp4", title="generated"):
    """Generate motion from a CLIP text embedding via guided latent diffusion.

    clip_embedding: (cond_dim,) or (B, cond_dim) tensor (the CLIP text features;
    encode text with the CLIP encoder in app/subpages/CLIP.py). The diffusion runs
    in standardized latent space, so we inverse-transform with the saved scaler
    before decoding.
    """
    model.eval().to(DEVICE)
    cond = clip_embedding.to(DEVICE).float()
    if cond.dim() == 1:
        cond = cond.unsqueeze(0)

    z = model.sample(cond, scale=cfg_scale)  # (B, latent_dim), standardized space
    if scaler is not None:
        z = torch.tensor(scaler.inverse_transform(z.cpu().numpy())).float().to(DEVICE)

    motion = decoder.decode(z)  # (B, T, 22, 3)
    plot_3d_motion_animation(motion[0].cpu().detach().numpy(), title,
                             figsize=(10, 10), fps=20, radius=2,
                             save_path=save_path, velocity=False)
    plt.close()
    return motion


def inference(model, decoder, text_to_clip, scaler=None, cfg_scale=2.5):
    """Interactive text->motion loop. ``text_to_clip`` maps a string to a CLIP
    embedding (cond_dim,)."""
    print('Inference mode (type "exit" to quit)')
    while True:
        text_input = input('Please enter a sentence: ')
        if text_input == 'exit':
            break
        emb = text_to_clip(text_input)
        predict(emb, model=model, decoder=decoder, scaler=scaler,
                cfg_scale=cfg_scale, title=text_input,
                save_path=f"recon_{text_input[:30].replace(' ', '_')}.mp4")
    print('Exiting Inference mode')
