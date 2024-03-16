# CONFIG FILE

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def upack_dict(d, out={}, prefix=""):
    """Recursively unpack dict"""
    for k, v in d.items():
        if isinstance(v, dict):
            out = upack_dict(v, out, prefix + k + ".")
        
        else:
            if not k.startswith("__"):
                if isinstance(v, list) or isinstance(v, tuple):
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            out = upack_dict(item, out, prefix + k + f".{i}.")
                        else:
                            out[prefix + k + f".{i}"] = item
                else:
                    out[prefix + k] = v
    return out

config = dotdict({
    
    # General
    ## network
    "MODEL": dotdict({
        'latent_dim': 12,
        'n_channels': [32, 32],
        'lin_size': 128,
        'n_linear': 3,
        'activation': 'leaky_relu',
        'dropout': 0.1,
        'batch_norm': True,
        'clip': False,  # not implemented
        # 'kernel_size': 2,  # can for now only be 2
        # 'stride': 2,  # can for now only be 2
        'learning_rate': 0.001,
        'optimizer': 'AdamW',
        'bool': True,

        'load_model' : False,
        'model_path' : '../tb_logs/MNISTAutoencoder/version_0/checkpoints/epoch=76-step=3542.ckpt',

        "LOSS" : dotdict({
            'mse': 1.,
            'klDiv': 0.000002,
            'l1': 0,
        }),
    }),

    'TRAINER' : dotdict({
        'max_epochs' : 100,
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
            'bool' : True,
        }),
    })
})
    

