import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import torch
from utils import test_translate
from utils import get_ckpt

import pytorch_lightning as pl
from modules.LatentMotionData import LatentMotionData
from modules.MotionLatentDiffusion import MotionLatentDiffusion
from utils import plot_3d_motion_animation, load_config

# logger
from pytorch_lightning.loggers import TensorBoardLogger

# get latent vectors
def find_saved_latent(path = f"logs/MotionVAE/VAE1/train/", cfg_name='config'):
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
    z_test = torch.load(path + '/latent_test.pt').to(torch.device('mps'))
    z_train = torch.load(path + '/latent_train.pt').to(torch.device('mps'))
    z_val = torch.load(path + '/latent_val.pt').to(torch.device('mps'))
    
    text_test = torch.load(path + '/clip_test.pt')
    text_train = torch.load(path + '/clip_train.pt')
    text_val = torch.load(path + '/clip_val.pt')

    file_num_test = torch.load(path + '/file_nums_test.pt')
    file_num_train = torch.load(path + '/file_nums_train.pt')
    file_num_val = torch.load(path + '/file_nums_val.pt')
    
    autoencoder = torch.load(path + '/model.pth').to(torch.device('mps'))
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
    def __init__(self, autoencoder, VAE_version):
        self.autoencoder = autoencoder
        self.VAE_version = VAE_version

    def decode(self, z):
        return decode_latent(z, self.autoencoder, self.VAE_version)

    def __call__(self, z):
        return self.decode(z)

def decode_latent(z, autoencoder, VAE_version):
    """
    makes a single vector in z into a reconstruction
    """
    autoencoder.eval()
    if VAE_version == 'VAE5':
        reconstruction = autoencoder.model.decode(z[0].unsqueeze(0), 
                                                torch.tensor([200]).to(torch.device('mps')))

    elif VAE_version == 'VAE1':
        # if VAE1
        reconstruction = autoencoder.decode(z[0].unsqueeze(0))

    elif VAE_version == 'VAE4':
        reconstruction = autoencoder.model.decode(z[0].unsqueeze(0), 
                                                torch.tensor([420]).to(torch.device('mps')))
        
    return reconstruction

# train
def train(VAE_version = 'VAE5'):

    # load config and instantiate logger
    cfg = load_config('motion_LD')
    logger = TensorBoardLogger('logs', name=f'MotionLD/{VAE_version}')


    # make animations folder
    if not os.path.exists(f'logs/MotionLD/{VAE_version}/animations'):
        os.makedirs(logger.log_dir + '/animations')
    # load latent vectors
    data_version, version = latent_picker(f'logs/MotionVAE/{VAE_version}/train/', cfg_name='hparams')
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
    decoder = LatentDecoder(autoencoder, VAE_version)

    # save decoder
    torch.save(decoder, logger.log_dir + '/decoder.pt')

    # model
    model = MotionLatentDiffusion(
        decode=decoder,
        scaler=scaler,
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


def predict(text_input, translate_inv, word2idx, model, decoder):
    # predict text input
    text_enc_input = translate_inv(text_input, word2idx).unsqueeze(0)
    noisy_latent = (torch.randn_like(z[0]) * 8.0).unsqueeze(0)
    print('text_enc_input', text_enc_input.shape)
    print('noisy_latent', noisy_latent.shape)
    # pred_noise, noise = model((noisy_latent.unsqueeze(0), text_enc_input.unsqueeze(0)))
    print(noisy_latent.sum())
    # # subtact pred
    t = 9
    out = noisy_latent.clone().to(torch.device('mps'))
    for i in range(t, 1, -1):
        print(i, end='\r')
        out = model._reverse_diffusion(out.to(torch.device('mps')),
                                    text_enc_input.to(torch.device('mps')),
                                    torch.tensor([i]).to(torch.device('mps')) )
        # print(noisy_latent.shape)

    print(noisy_latent.sum())
    print(out.sum())
    # decode
    reconstruction = decoder.decode(out)
    recon_noisy = decoder.decode(noisy_latent.to(torch.device('mps')))
    print(reconstruction.shape)

    plot_3d_motion_animation(reconstruction[0].cpu().detach().numpy(), text_input,
                                            figsize=(10, 10), fps=20, radius=2, save_path=f"recon_text.mp4", velocity=False)
    plt.close()

    plot_3d_motion_animation(recon_noisy[0].cpu().detach().numpy(), text_input,
                                            figsize=(10, 10), fps=20, radius=2, save_path=f"recon_text_noisy.mp4", velocity=False)
    plt.close()

def inference(trans_inv, word2idx, model, decoder):
    # inference mode
    print('Inference mode')
    while True:
        text_input = input('Please enter a sentence: ')
        if text_input == 'exit':
            break
        predict(text_input, 
                translate_inv=trans_inv,
                word2idx=word2idx,
                model=model,
                decoder=decoder)
    print('Exiting Inference mode')
