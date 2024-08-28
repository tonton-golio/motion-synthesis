import streamlit as st


def render():
    # Title and introduction
    st.write("""
    # CLIP

    *Contrastive Language-Image Pairs (CLIP) is a neural network that learns to associate images and text.*
    """)

    cols = st.columns(2)
    with cols[1]:
        st.image('app/assets_produced/17_CLIP/dalle_woman_on_boat.png', caption='dalle prompt: Give me an image containing a woman on a boat in a lake, with a mountain and the sky in the bacground.')

    with cols[0]:
        st.write("""
        Given the image on the right, a classification model, might output a list of classes,

        [`woman`, `boat`, `lake`, `mountain`, `sky`].

        This approach does not scale well, as we cannot possibly list **all** classes. Instead, we could try to do captioning:
        
        `A woman in a boat, on a lake, with a mountain and the sky in the background.`

        This gives nicer descriptions, and scales better, but it is still limited by the captions we can scrape from the internet.

        Instead what CLIP does, is embed images and text into the same space, a joint embedding. In the text space, interpolation is done very well by transformers, and since our space is unified, we can trust the model to generate unseen class-combinations. This is demonstrated in \cite{zhang2022contrastive}, where we see great performance of CLIP on hand drawn sketches of objects, which are unseen class-combinations. This tells us that CLIP is able to learn the abstraction of an object, and not just the object itself.
        """)


    st.write(r"""
    

    CLIP is trained on 400M image-caption pairs, and the way it is trained is that we have a VisionTransformerEncoder and a TextTransformerEncoder, we pass them and image and a text respectively, and train the model to maximize the dot product between the two embeddings, i.e., we want them to be close in the joint space. Additionally, we train the model to minimize the dot product between the image and all other texts, and the text and all other images.

    A batch of $n$ images and $n$ texts, yields $2n$ vectors, $p$ and $q$, respectively, both of latent dimension, $d$. We compute the dot product between all pairs of vectors, to build the similarity matrix, $S$, I would expect square root of the dimension, $d$, in the denominator included for variance stabilization, similarly to the softmax-scaling in the transformer.
    """)
    st.write(r"""
    $$
        S_{i,j} = \frac{p_i\cdot q_j^T}{\sqrt{d}},
    $$

    But instead what CLIP does, is use the cosine similarity, this takes out dependence on vetor length, only considering angle difference
    $$
        S_{i,j} = \frac{p_i\cdot q_j^T}{\|p_i\|\|q_j\|}.
    $$

    The trace has our correctly matched pairs, and the rest of the matrix has our negative samples. So we maximize the trace of $S$, and minimize the rest of the matrix.

    Does this mean batch size should be kept small; such that it is unlikely to have, say two images of cats in a batch?

    Using CLIP for our motion transformer should work. We will only be using the text encoding part of the model, then we will feed this to our motions latent diffusion model,
    """)

    st.write("""
    We also see texts and image connected for the case of xray scans of lungs, in the paper: Contrastive Learning of Medical Visual Representations from Paired Images and Text (https://arxiv.org/abs/2010.00747) (ConVIRT)


    CLIP also has a zero-shot classification ability, but this is not what we will be using it for in our project.
    """)

    st.write("""
    a huge classification model does not scale, as we can not possibly five
    Instead of training 

    It is trained on a large dataset of images and their captions, and learns to predict which caption corresponds to which image. This allows it to perform a variety of tasks, such as zero-shot classification, image generation, and more.
    """)


    # Sources
    st.write("""
    ### Sources
    - [Computerphile video](https://www.youtube.com/watch?v=KcSXcpluDe4)

    """)




    st.divider()

    """
    ## Section 2: CLIP in the context of the Motion Transformer

    So our text strings describe a motion sequence, so the still (single-frame) world-understanding may not be sufficient, but lets try:
    """

    st.code("""
    from transformers import CLIPProcessor, CLIPModel


    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    text = "a person is walking"
    inputs = processor(text, return_tensors="pt", padding=True)
    outputs = model(**inputs)
            """)
    

    st.divider()

    st.image('app/assets_produced/17_CLIP/label_similarity.png', caption='Label similarity')
    st.image('app/assets_produced/17_CLIP/class_similarity.png', caption='Class similarity')
    st.image('app/assets_produced/17_CLIP/label_projection.png', caption='Label projection')