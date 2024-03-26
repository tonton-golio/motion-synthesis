from os.path import join as __pjoin
import sys; sys.path += ['/Users/tonton/Documents/motion-synthesis']
from global_utils import dotdict

# this is a dictionary of hyperparameters
config = dotdict({

    "MODEL" : dotdict({
        "hidden_dim": 1024,
        "hidden_dim_trans" : 1024,
        "n_layers": 5,
        "n_heads": 6,
        "dropout": 0.1,
        "latent_dim": 1024,
        "learning_rate": 1 * 1e-8,
        "optimizer": "AdamW",
        "_save_animations": True,
        "load": True,
        "_checkpoint_path": 'latest',#"../tb_logs3/TransformerMotionAutoencoder/version_9/checkpoints/epoch=33-step=1122.ckpt",
        'output_layer' : 'linear', ##"linear", # or "transformer" or None
        'activation' : "relu",
        'transformer_activation' : "relu", # or #gelu
        'clip_grad_norm': 0.1,
        'batch_norm': True,
        
        'loss_weights' : dotdict({
            'kl_div': 2*1e-8,
            'velocity_relative' : .5,
            'root' : .01,
            'pose0' : 0.01,
            'motion' : 0.,
            'motion_relative' : 2.,
        }),

    }),

    "DATA" : dotdict({
        "seq_len": 160,
        "batch_size": 128,
        "file_list_paths": {
            "_train": "../../data/HumanML3D/HumanML3D/train_cleaned.txt",
            "_val": "../../data/HumanML3D/HumanML3D/val_cleaned.txt",
            "_test": "../../data/HumanML3D/HumanML3D/test_cleaned.txt",
        },
        "_motion_path": "../../data/HumanML3D/HumanML3D/new_joints",
    }),

    'TRAINING' : dotdict({
        "max_epochs": 300,
        "accelerator": "mps",
        "devices": 1,
        "precision": "32-true",
        "n_gpus": 1,
        "fast_dev_run": False,
    }),
})

