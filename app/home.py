import streamlit as st
import base64

title = "# Motion: synthesized from text"
abstract = """
    Motion synthesis is a fast evolving field, with groundbreaking research being published every year. The task is to synthesise humanoid motion from a text prompt. I am implementing such a model, and will demonstrate/explain it here.

    The approach I will be employing, is diffusion on a latent representation. This is a powerful technique demonstrated in the literature. I work with the publicly available HumanML3D dataset, which contains 3D motion capture data with descriptive text strings from KIT.
    """

def todo():
    st.write("""
    ## To do
    - [] Renew thesis contract
        - [x] Email supervisors
        - [x] Send contract (30 may 2024)
    - [] Implement metrics for image space diffusion
    - [] Motion latent diffusion
        - [x] make VAE save latent space at test end
            - [x] VAE1
            - [x] VAE4
            - [x] VAE5
        - [x] Refactor latent diffusion notebook
        - [x] Implement latent diffusion in main.py
    - [x] Refactor motion VAEs such that they all have the same style.
    - [] Compare pose VAEs
        - [x] implement CONV based VAE
        - [x] refactor VAEs to have same style
        - [] tune each model
    - [] Add pauls notes to manuscript
    - [x] merge utils and utils_pose
    - [x] merge pose_trainer and main
    - [] plot pose distribution (if we take each pose, center it, and rotate it to the same orientation (using the hip)
    - [] balance pose dataset
    - [] to validate pose model, try to VAE all frames of a seq, and then output a seq.
    - [x] clean up MotionData.py (delete 1 module)
    - [x] merge motion VAE config files
    - [] take everything from motion/scripts and put in main.py and utils.py
    - [] implement wrapper in app.py: if not image found, produce and save image. +re run button
    """)

def home():
    # introduction and flow
    cols = st.columns([6, 1])
    with cols[0]:  # Introduction
        st.markdown(abstract)

    cols[1].image('assets/0_home/flow_pipeline/flow2.png', caption='Schematic of the architecture/flow employed.', width=180)  # Flow

    # model interface
    ## text prompt
    cols[0].text_area('Text prompt', 'A person turns on the spot.')

    ## centering the video.
    st.columns([1, 2, 1])[1].video('assets/0_home/recon_fake.mp4')

def learning_goals():
    st.write("""
    My goals in this project we the following:
    * Acquire deep knowledge of the transformer architecture, and diffusion model.
    * Learn how to build a large AI project from scratch. How to manage the codebase, and the project as a whole. And which modern tools to use (e.g. PyTorch Lightning, Tensorboard, Hydra).
    * Implement a model that can generate motion from a text prompt.
    """)

def embed_pdf(pdf_file='assets/papers/Thesis_compressed_compressed.pdf', st=st, height=700, width=700):
    with open(pdf_file, 'rb') as f:
        pdf = f.read()#).decode()
    base64_pdf = base64.b64encode(pdf).decode()
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="{width}" height="{height}" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

if __name__ == '__main__':
    st.set_page_config(page_title='Motion Synthesis from a Text Prompt', page_icon='🕺')
    st.markdown(title)

    tab_names = ['TODO', 'Home', 'Learning Goals', 'Thesis']

    tabs = {name: tab for name, tab in zip(tab_names, st.tabs(tab_names))}

    with tabs['TODO']:
        todo()
    with tabs['Home']:
        home()
    with tabs['Learning Goals']:
        learning_goals()
    with tabs['Thesis']:
        embed_pdf()
