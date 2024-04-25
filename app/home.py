import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import base64
# from st_pages import show_pages_from_config
# show_pages_from_config(".streamlit/pages_sections.toml")




def embed_pdf(pdf_file, st=st, height=500, width=700):
    with open(pdf_file, 'rb') as f:
        base64_pdf = base64.b64encode(f.read()).decode()
    # base64_pdf
    pdf_display1 = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    pdf_display1
    pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="{width}" height="{height}" type="application/pdf">'
    pdf_display

    st.markdown(pdf_display, unsafe_allow_html=True)
def display_paper_tabs(path_prefix = 'assets/papers/'):
    # make tabs for the four different papers I haver written on the topic
    paths = {
        
        'motion dataset' : 'Motion_Data.pdf',
        'motion autoencoder' : 'Motion_Latent_Embedding.pdf',
        'motion diffusion' : 'Motion_diffusion.pdf',
        'mnist autoencoder' : 'VAE_Optimization_for_MNIST.pdf',
        'mnist diffusion' : 'Diffusion_MNIST.pdf',
        'thesis' : 'Thesis_compressed.pdf'
    }

    # "## My papers"

    # tabs = st.tabs(list(paths.keys()))

    # for i, tab in enumerate(tabs):
    #     with tab:
    #         try:
    #             # pdf_viewer(path_prefix+paths[list(paths.keys())[i]], key=list(paths.keys())[i])
    #             embed_pdf(path_prefix+paths[list(paths.keys())[i]])
    #         except:
    #             st.write('Not yet available')

    "## Thesis: Text-conditioned Reverse Diffusion on Latent Representation of Humanoid Motion"
    embed_pdf(path_prefix+paths['thesis'])
# title
"""
# Motion Synthesis
"""

# introduction and flow
cols = st.columns([6, 1])
with cols[0]:
    """
    Motion synthesis is a fast evolving field, with groundbreaking research being published every year. The task is to synthesis humanoid motion from a text prompt. I have implemented such a model, and will demonstrate/explain it here.

    The approach I will be employing, is diffusion on a latent representation. This is a powerful technique demonstrated in the literature
    . I work with the publicly available HumanML3D dataset, which contains 3D motion capture data and descriptive text strings.

    Below you see a schematic of the architecture/flow employed.
    """

with cols[1]:
    # include image showing flow
    path = 'assets/flow2.png'
    st.image(path, caption='Schematic of the architecture/flow employed.', width=180)

# model interface


# in cols 0 we want a text input space for the user to input a text prompt
with cols[0]:
    text = st.text_area('Text prompt', 'A person turns on the spot.')

# # in cols 1 we want to display the motion generated from the text prompt
cols = st.columns([1, 2, 1])
with cols[1]:
    # include video of generated motion
    path = 'assets/recon_0.mp4'
    st.video(path)


# include papers
display_paper_tabs()

