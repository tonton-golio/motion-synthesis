# CONFIG FILE
import sys; sys.path += ['/Users/tonton/Documents/motion-synthesis']
from global_utils import dotdict
config = dotdict({
    
    # General
    ## network
    "MODEL": dotdict({
        'latent_dim': 12,
        'n_channels': [128, 128],
        'lin_size': 512,
        'n_linear': 3,
        'activation': 'leaky_relu',
        'dropout': 0.1,
        'batch_norm': True,
        'clip': False,  # not implemented
        # 'kernel_size': 2,  # can for now only be 2
        # 'stride': 2,  # can for now only be 2
        'learning_rate': 0.0005,
        'optimizer': 'AdamW',
        'bool': False,

        'load_model' : False,
        'model_path' : '../tb_logs/MNISTAutoencoder/version_0/checkpoints/epoch=76-step=3542.ckpt',

        "LOSS" : dotdict({
            'mse': 1.,
            'klDiv': 0.0000001,
            'l1': 0,
        }),
    }),

    'TRAINER' : dotdict({
        'max_epochs' : 20,
    }),

    # Data
    'DATA' : dotdict({
        '__path' : '../../data/other_data',
        'batch_size' : 1024,
        '__num_workers' : [4, 2, 1], # train, val, test
        '__include_digits' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'transforms' : dotdict({
            'rotate_degrees' : 10,
            'distortion_scale' : 0.3,
            'translate' : (0, 0),
            'normalize' : (0.1307, 0.3081),
            'shear' : 0.3,
            'scale' : .3,
            'bool' : False,
        }),
    })
})
    

