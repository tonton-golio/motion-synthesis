# CONFIGURATION FILE
import sys; sys.path += ['/Users/tonton/Documents/motion-synthesis']
from global_utils import dotdict
config = dotdict({
    
    # General
    ## network
    "MODEL": dotdict({
        'timesteps': 100,
        'time_embedding_dim': 12,
        'target_embedding_dim': 4,
        'noise_multiplier': 4.0,
        'hidden_dim': 512,
        'n_hidden': 6,
        'activation': 'leaky_relu',
        'dropout': 0.1,
        'batch_norm': True,
        'clip': False,  # not implemented
        'learning_rate': 0.0005,
        'optimizer': 'AdamW',
        'epsilon': 1e-5,

        'load_model' : False,
        'model_path' : '../tb_logs3/LatentDiffusionMNIST/version_10/checkpoints/epoch=49-step=15000.ckpt',

        'loss' : 'mse',
    }),

    'TRAINER' : dotdict({
        'max_epochs' : 20,
    }),

    # Data
    'DATA' : dotdict({
        '__path' : '../../data/other_data',
        'batch_size' : 128,
        'N' : 48000,
        'V' : 7,

        }),
    })