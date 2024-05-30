import streamlit as st
import numpy as np
import os, sys, glob
import matplotlib.pyplot as plt
# path join 
from os.path import join as pjoin
from pathlib import Path

# Intro and title
"""
# Motion Autoencoder


"""
tab_names = [
    'VAE1',
]
tabs = {name: tab for name, tab in zip(tab_names, st.tabs(tab_names))}


with tabs['VAE1']:
    st.write("""
    Our first model, VAE1, is a transformer based VAE which uses linear layers for compression and decompression.
             """)