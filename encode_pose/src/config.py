from os.path import join as __pjoin

###### TRACKED PARAMETERS ######
# data parameters
BATCH_SIZE = 1024

# model parameters
N_LAYERS = 4
N_LIN_LAYERS = 6
N_LIN_UNITS = 512
LATENT_DIM = 48
DROPOUT = 0.10
CHANNELS_HIDDEN = 64
CHANNELS_OUT = 32
MODEL_TYPE = "graph"

# training parameters
EPOCHS = 20
LEARNING_RATE =  1* 1e-3
OPTIMIZER = "AdamW"
KL_WEIGHT = .00000001
load = False
checkpoint_path = "latest"


###### UNTRACKED PARAMETERS ######
# paths
__base_path = "../../data/HumanML3D/HumanML3D/"
__sets = ["train", "val", "test"]
__FILE_LIST_PATHS = {i: __pjoin(__base_path, f"{i}_cleaned.txt") for i in __sets}
__MOTION_PATH = __pjoin(
    __base_path,
    "new_joints",
)


# device parameters
__ACCERLATOR = "mps"
__DEVICES = 1
__PRECISION = "32-true"  #"16-mixed" # does not work with mps
__N_GPUS = 1
__N_WORKERS = 4

# debugging
__FAST_DEV_RUN = False
# __SAVE_ANIMATIONS = True

# metric
# how do i get a metric for hparams in tensorboard?
# https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html#tensorboard
