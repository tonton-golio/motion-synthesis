from os.path import join as __pjoin

###### TRACKED PARAMETERS ######
# data parameters
SEQ_LEN = 160 # 200
BATCH_SIZE = 128

# model parameters
N_LAYERS = 4
N_HEADS = 6
LATENT_DIM = 256
DROPOUT = 0.10
HIDDEN_DIM = 1024

# training parameters
EPOCHS = 100
LEARNING_RATE = 1 * 1e-4
OPTIMIZER = "AdamW"
LOSS_FUNCTION = "MSELoss + KL"
KL_WEIGHT = .00001


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
__SAVE_ANIMATIONS = True

# metric
# how do i get a metric for hparams in tensorboard?
# https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html#tensorboard
