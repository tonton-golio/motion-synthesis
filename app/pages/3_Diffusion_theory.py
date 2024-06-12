import streamlit as st

st.set_page_config(
    page_title="Diffusion Theory",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.write("# Diffusion Theory")
tab_names = [
    'Introduction',
    'Measuring noise-level',
    'Noise Schedule',
    'Time Embedding',
    'Metrics',
]
tabs = {name: tab for name, tab in zip(tab_names, st.tabs(tab_names))}

# Introduction
with tabs['Introduction']:
    from subpages.diffusion_intro import diffusion_intro
    diffusion_intro()

# Measuring noise-level
with tabs['Measuring noise-level']:
    from subpages.measure_noise_level import measure_noise_level_page
    measure_noise_level_page()

# Noise Schedule
with tabs['Noise Schedule']:
   from subpages.noise_schedule import noise_schedule_page
   noise_schedule_page()

# Time Embedding
with tabs['Time Embedding']:
    from subpages.time_embedding import time_embedding_page
    time_embedding_page()

# Metrics
with tabs['Metrics']:
    from app.subpages.metrics_diffusion import metrics_diffusion_page
    metrics_diffusion_page()

