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




    




