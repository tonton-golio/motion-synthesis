# Motion statistics
# Author: Anton Golles
# This script motion data for the following properties:
#   - Shape consistency
#   - Velocity
#   - Nan and zero values
# and then trims the data accordingly



import streamlit as st
import numpy as np
import os, sys, glob
import matplotlib.pyplot as plt
# path join 
from os.path import join as pjoin
from pathlib import Path

# Intro and title
"""
# Motion Data


"""
tab_names = [
    'Motion statistics',
    'Text statistics',
]
tabs = {name: tab for name, tab in zip(tab_names, st.tabs(tab_names))}


# Motion statistics

# load
def find_files(directory, extension):
    """Find all files in a directory with a specific extension."""
    return list(Path(directory).glob(f'*.{extension}'))

def load_npy_file(file_path):
    """Safely load a .npy file, returning None if an error occurs."""
    try:
        return np.load(file_path)
    except Exception as e:
        return None

def file_name_change(file, to_short=True):
    if to_short:
        
        # if its PosixPath change to string and the cut of the folder and extension
        # else just cut of the folder and extension
        return str(file).split('/')[-1].split('.')[0]

    else:
        # make it into the revelant PosixPath
        raise NotImplementedError

# checks
def shape_consistency_check(files, min_frames=10, verbose=True, plot=False, **kwargs):
    """Check the shape consistency of .npy files and identify files with fewer than min_frames."""
    #if verbose: logging.info('Checking shape consistency...')
    valid_shapes, bad_files = [], []
    
    for file in files:
        data = load_npy_file(file)
        if data is None or data.shape[0] < min_frames or len(data.shape) != 3:
            bad_files.append(file)
        else:
            valid_shapes.append(data.shape[0])


    fig = None
    if plot:
        # print('Making plot')
        fig = plt.figure(figsize=(4, 3))
        plt.hist(valid_shapes, bins=20, edgecolor='black', linewidth=1.2, facecolor='salmon', alpha=0.7)
        plt.title('Distribution of sequence lengths')
        plt.xlabel('Sequence Length')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.tight_layout()
        # plt.savefig('assets/sequence_length_distribution.png')

    
    return [file.name for file in bad_files], fig

def check_velocity(data, threshold=0.5, root_idx = 0):
    """Check if the velocity of the root joint exceeds a threshold."""
    # print(data.shape)
    root_traversal = data[:, root_idx, :]
    # print(root_traversal.shape)
    # print(root_traversal.shape)
    velocity = np.linalg.norm(np.diff(root_traversal, axis=0), axis=1)
    return (velocity > threshold).any(), velocity

def check_velocity_all(files, threshold=.5, root_idx = 0, plot=False, **kwargs):
    """Check if the velocity of the root joint exceeds a threshold for all files."""
    bad_files = []
    velocities = {'good': [], 'bad': []}
    for file in files:
        data = load_npy_file(file)
        if data is not None:
            exceeds_threshold, velocity = check_velocity(data, threshold, root_idx)
            if exceeds_threshold:
                bad_files.append(file)
                velocities['bad'].append(velocity)
            else:
                velocities['good'].append(velocity)

    velocities['all'] = velocities['good'] + velocities['bad']
    for k, v in velocities.items():
        velocities[k] = np.hstack(v, ) if len(v) > 0 else np.array([])

    fig = None
    if plot:
        fig, ax = plt.subplots(1,2, figsize=(8, 3))
        ax[0].hist(velocities['all'], bins=50, edgecolor='black', linewidth=1.2, facecolor='salmon', alpha=0.7)
        ax[1].hist(velocities['good'], bins=50, edgecolor='black', linewidth=1.2, facecolor='salmon', alpha=0.7)
        for a in ax:
            a.set_yscale('log')
            a.set_xscale('log')
            a.set_ylabel('Frequency')
            a.set_xlabel('Velocity')
                
        ax[0].axvline(threshold, c='black', linestyle='--')
        fig.suptitle('Velocity of root joint')

        plt.tight_layout()
        # plt.savefig('assets/velocity_of_root_joint.png')



    return bad_files, fig

def nan_zero_check(files, verbose=False, plot=True, **kwargs):
    if verbose: print('-----------------------\nChecking for nan and zero values:')
    # check for nan values or zeros
    nan_count = 0
    zero_count = 0
    bad_files = []
    for f in files:
        data = np.load(f)
        nan_count_inc = np.isnan(data).sum() > 0
        zero_count_inc = (data == 0).sum() > 3
        if nan_count_inc:
            nan_count += 1
            bad_files.append(f)
        if zero_count_inc:
            zero_count += 1
            bad_files.append(f)

    if verbose:
        print('Number of nan values:', nan_count)
        print('Number of zero values:', zero_count)
        print('Files with nan or zero values:', bad_files)
        print('-----------------------')
    fig = None
    if plot:
        pass

    return bad_files, fig

# remove files
def remove_from_lst(files, bad_files):            
    len_before = len(files)
    files = [str(f).split('/')[-1].split('.')[0] for f in files]
    # print(files)
    bad_files = [str(f).split('/')[-1].split('.')[0] for f in bad_files]
    # print(bad_files)
       
    files_reduced = [f for f in files if f not in bad_files]

    files_reduced = [pjoin(path, f+'.npy') for f in files_reduced]
    # len_after = len(files_reduced)    
    return files_reduced#, len_before - len_after

def run_check(func, files, plot=False, **kwargs):
    bad_files, fig = func(files, plot=plot, **kwargs)
    files = remove_from_lst(files, bad_files)
    print('After check:', func.__name__, len(files))
    return files, fig

#  train test val
def get_train_test_val(base_path = 'stranger_repos/HumanML3D/HumanML3D/', suffix = ''):
    # load text file
    paths = {i : f'{base_path}{i}{suffix}.txt' for i in ['train', 'val', 'test']}
    files_4_selection = {k : np.loadtxt(v, delimiter=',', dtype=str) for k, v in paths.items()}
    return files_4_selection

def trim_files_4_selection(files_4_selection, files):
    files_4_selection = files_4_selection.copy()
    for k, v in files_4_selection.items():
        files_4_selection[k] = [f for f in v if f in files]
    return files_4_selection

def print_summary(files_4_selection, title='Summary'):
    lengths = {k : {k1 : len(files_4_selection[k][k1]) for k1 in files_4_selection[k]} for k in files_4_selection}
    
    # include sum
    for k in lengths:
        lengths[k]['sum'] = sum(lengths[k].values())
    # print(lengths)
    keys = list(lengths[list(lengths.keys())[0]].keys())

    text = title.title()+':'+'\n'
    
    text += f' '*16 + '\t'.join([k for k in keys]) + '\n'
    for k, v in lengths.items():
        text += f'{str(k).rjust(10)}:\t' + '\t'.join([str(v[k1]) for k1 in keys]) + '\n'

    print(text)
    
with tabs['Motion statistics']:
    base_path = '../stranger_repos/HumanML3D/HumanML3D/'
    path = pjoin(base_path,  'new_joints/')
    files = find_files(path, 'npy')
    print('After get filelist: len(files)' , len(files))
    # run checks
    for check in [shape_consistency_check, nan_zero_check, check_velocity_all]:
        files, fig = run_check(check, files, plot=True, threshold=0.5, root_idx=0)
        if fig is not None:
            st.pyplot(fig)
    # get train test val
    files = [file_name_change(f, to_short=True) for f in files]
    files_4_selection = {'raw': get_train_test_val(base_path)}
    files_4_selection['trimmed'] = trim_files_4_selection(files_4_selection['raw'], files)
    print_summary(files_4_selection)
    files_4_selection = files_4_selection['trimmed']

    # save new files
    for k, v in files_4_selection.items():
        np.savetxt(f'{base_path}{k}_cleaned.txt', v, delimiter=',', fmt='%s')


# Text statistics
with tabs['Text statistics']:
    import streamlit as st
    import os
    import tiktoken

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    import numpy as np
    import matplotlib.pyplot as plt

    @st.cache_data
    def load_texts(file_name='HumanML3D/HumanML3D/texts/000006.txt'):
        with open(file_name, "r") as f:
            texts = f.read().split("\n")
            texts = [text for text in texts if text]  # remove empty strings
        return texts

    @st.cache_data
    def get_encodings():
        enc = tiktoken.encoding_for_model("gpt-4")
        path_0 = 'HumanML3D/HumanML3D/'
        files = os.listdir(path_0 + 'texts/')[:100]
        data = {}
        idx = 0
        max_len = 0
        for file in files:
            texts = load_texts(path_0 + 'texts/' + file)
            texts_enc = [enc.encode(text) for text in texts]
            data[idx] = {
                "file": file,
                "texts": texts,
                "texts_enc": texts_enc
            }

            max_len = max(max_len, max([len(t) for t in texts_enc]))

            idx += 1
        return data, max_len


    # To get the tokeniser corresponding to a specific model in the OpenAI API:
    @st.cache_data
    def get_data():
    

        data, max_len = get_encodings()
        "max_len:", max_len

        # dataset
        for i, d in data.items():
            data[i]["texts_enc"] = [t + [0] * (512 - len(t)) for t in d["texts_enc"]]



            # len(data[i]["texts_enc"][0]), 1

        # #plot data shapes (histogram)
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # plt.hist([len(d["texts_enc"]) for d in data.values()])
        # st.pyplot(fig)
            
        data = {k: v for k, v in data.items() if len(v["texts_enc"]) == 3}
        data = np.stack([d["texts_enc"] for d in data.values()], axis=0)
        st.write('data.shape:', data.shape)

        # set of all tokens
        vocab_size = 100194+1 #np.unique(data.flatten()).shape[0]


        return data, vocab_size

    class TextDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            texts_enc = self.data[idx]
            text_enc = texts_enc[0]
            return torch.tensor(text_enc)
        

    @st.cache_data
    def get_loader(data):
        dataset = TextDataset(data)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        return dataloader

    data, vocab_size = get_data()

    class TextTransformerAutoencoder(nn.Module):
        def __init__(self, max_len, vocab_size):
            super().__init__()
            self.max_len = max_len
            self.vocab_size = vocab_size
            self.embedding = nn.Embedding(vocab_size, 128)
            self.anti_embedding = nn.Linear(128, vocab_size)
            self.transformer = nn.Transformer(
                d_model=128, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512, dropout=0.1
            )

            self.linear = nn.Linear(128*512, 10)

            self.linear2 = nn.Linear(10, 128*512)

            self.transformer_decoder = nn.Transformer(
                d_model=128, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512, dropout=0.1
            )

        def anti_embed(self, x):
            return self.anti_embedding(x)

        def forward(self, x, verbose=False):
            x = self.embedding(x)
            embedding = x.clone()
            if verbose: st.write(x.shape)
            x = self.transformer(x, x)
            x = nn.Flatten()(x) # flatten the output
            if verbose: st.write(x.shape)
            x = self.linear(x)
            if verbose: st.write(x.shape)
            x = self.linear2(x)
            if verbose: st.write(x.shape)
            x = x.view(-1, 512, 128)
            if verbose: st.write(x.shape)
            x = self.transformer_decoder(x, x)
            if verbose: st.write(x.shape)

            return x, embedding


        

    n_epochs = 3
    model = TextTransformerAutoencoder(32, vocab_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataloader = get_loader(data)

    c = st.empty()
    for epoch in range(n_epochs):
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            x = batch
            y, embedding = model(x, verbose=False)
            loss = criterion(y, embedding)
            loss.backward()
            optimizer.step()
            c.text(f"epoch: {epoch}, batch: {i}, loss: {loss.item()}")




        





