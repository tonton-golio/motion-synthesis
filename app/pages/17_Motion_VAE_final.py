import streamlit as st


# Title and Introduction
"""
# Motion VAE - Final results

We obtain a latent embedding from our `Motion VAE` model. If we take the actions optained in the text shortening step, we actions are grouped together and similar action are near each other. This is shown in the Figure below
"""

st.image('assets/17_Motion_VAE_final/actions_highlighted.png')

"""
Specifically, notice that the action `walk` occupies the top-left region. The action `sidestep` is also here. At the same time, we see actions like `wave`, `throw` and `clap` grouped together on the right side. This is a good sign that the model has learned to group similar actions together.


If we instead use the action categories, we see a similar coherence in the latent space. This is shown in the Figure below
"""

st.image('assets/17_Motion_VAE_final/groups_highlighted.png')


"""
This gives us confidence, that the `Motion VAE` model has learned a sufficiently good latent representation of the actions. Thus we should be ready to train a diffusion model on this latent space.
"""