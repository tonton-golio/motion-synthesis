from os.path import join as __pjoin
import sys; sys.path += ['/Users/tonton/Documents/motion-synthesis']
from global_utils import dotdict

# this is a dictionary of hyperparameters
config = dotdict({

    "MODEL" : dotdict({
        "input_length": 200,
        "input_dim": 66,
        "hidden_dim": 512,
        "n_layers": 2,
        "n_heads": 6,
        "dropout": 0.1,
        "latent_dim": 256,
        "LOSS" : dotdict({
            'mse': 1.,
            'klDiv': 0.000001,
            'l1': 0,
        }),
        "learning_rate": 1 * 1e-4,
        "optimizer": "AdamW",
        "save_animations": True,
        "load": False,
        "checkpoint_path": "../tb_logs/TransformerMotionAutoencoder/version_28/checkpoints/epoch=9-step=1290.ckpt",
        'output_layer' : "linear", # or "transformer" or None
        'activation' : "leaky_relu",
        'transformer_activation' : "gelu",
    }),

    "DATA" : dotdict({
        "seq_len": 200,
        "batch_size": 256,
        "file_list_paths": {
            "train": "../../data/HumanML3D/HumanML3D/train_cleaned.txt",
            "val": "../../data/HumanML3D/HumanML3D/val_cleaned.txt",
            "test": "../../data/HumanML3D/HumanML3D/test_cleaned.txt",
        },
        "motion_path": "../../data/HumanML3D/HumanML3D/new_joints",
    }),

    'TRAINING' : dotdict({
        "max_epochs": 10,
        "accelerator": "mps",
        "devices": 1,
        "precision": "32-true",
        "n_gpus": 1,
        "fast_dev_run": False,
    }),
})

