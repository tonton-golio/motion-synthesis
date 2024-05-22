import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import base64

# from st_pages import show_pages_from_config
# show_pages_from_config(".streamlit/pages_sections.toml")

def embed_pdf(pdf_file, st=st, height=700, width=700):
    with open(pdf_file, 'rb') as f:
        pdf = f.read()#).decode()
    base64_pdf = base64.b64encode(pdf).decode()
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="{width}" height="{height}" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# title
"""# Motion Synthesis from Text Prompt"""

# introduction and flow
cols = st.columns([6, 1])
with cols[0]:  # Introduction
    """
    Motion synthesis is a fast evolving field, with groundbreaking research being published every year. The task is to synthesise humanoid motion from a text prompt. I am implementing such a model, and will demonstrate/explain it here.

    The approach I will be employing, is diffusion on a latent representation. This is a powerful technique demonstrated in the literature. I work with the publicly available HumanML3D dataset, which contains 3D motion capture data with descriptive text strings from KIT.
    """

cols[1].image('assets/0_home/flow_pipeline/flow2.png', caption='Schematic of the architecture/flow employed.', width=180)  # Flow

# model interface
## text prompt
cols[0].text_area('Text prompt', 'A person turns on the spot.')

## centering the video.
st.columns([1, 2, 1])[1].video('assets/0_home/recon_fake.mp4')

# include thesis
embed_pdf('assets/papers/Thesis_compressed_compressed.pdf')

