import streamlit as st
from utils_app import load_or_save_fig, VarianceSchedule, kl_score
from subpages.diffusion_intro import prep_image
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec

deactivate = False

# Noise metrics
def normalize_01(vector):
    vector -= vector.min()
    vector /= vector.max()
    return vector

def shannon_entropy_binned(vector):
    if isinstance(vector, torch.Tensor):
        vector = vector.clone().detach()
    vector = normalize_01(vector)
    # Compute the histogram of the vector
    bins=int(np.sqrt(len(vector)) + 1)
    counts, _ = np.histogram(vector, bins=bins)
    # Filter out zero counts
    counts = counts[counts > 0]
    # Calculate probabilities
    probabilities = counts / len(vector)
    # Compute Shannon entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def shannon_entropy_2d(im, return_all=False):
    im_arr = np.array(im)
    # im_arr.shape
    im_derivative = np.gradient(im_arr)
    # st.write(im_derivative[0].shape)
    # im_derivative[0].shape
    im_derivative = im_derivative[0]**2 + im_derivative[1]**2
    # im_derivative /= im_derivative.sum()
    # st.write(im_derivative.shape)
    
    # H = -np.sum(im_derivative[im_derivative > 0] * np.log2(im_derivative[im_derivative > 0]))
    H = shannon_entropy_binned(im_derivative.flatten())
    if return_all:
        return H, im_derivative
    return H

# demos
#@load_or_save_fig('assets_produced/3_Diffusion_theory/shannon_entropy_scale_dependence_plot.png', deactivate=deactivate)
def shannon_entropy_scale_dependence_plot():
    vec_lengths = np.logspace(2, 6, 50, dtype=int)
    H1_size = [shannon_entropy_binned(torch.randn(length),
                                    ) for length in vec_lengths]

    fig, ax = plt.subplots(2, 1, figsize=(5, 8))
    ax[0].scatter(vec_lengths, H1_size, marker='x', label='Shannon entropy (H1)', color='orangered')
    ax[0].set_title('Shannon entropy of Gaussian noise vectors of different lengths')
    ax[0].set_xscale('log')
    ax[0].set_xlabel('Vector length')
    ax[0].set_ylabel('H1')

    # fit with log

    func = lambda x, a,b,c: a*np.log2(x-c) + b
    x = np.linspace(min(vec_lengths), max(vec_lengths), 100)
    from scipy.optimize import curve_fit
    popt, _ = curve_fit(func, vec_lengths, H1_size)
    y = func(x, *popt)
    ax[0].plot(x, y, label=f'fit: {popt[0]:.2f}log2(x-{popt[2]:.2f}) + {popt[1]:.2f}', color='purple')
    ax[0].legend()


    H1_size_normalized_by_length = np.array(H1_size) / func(vec_lengths, *popt)

    # fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax[1].scatter(vec_lengths, H1_size_normalized_by_length, marker='x', label='Shannon entropy (H1) normalized by length', color='orangered')
    ax[1].set_title('Shannon entropy of Gaussian noise vectors of different lengths')
    ax[1].set_xscale('log')
    ax[1].set_xlabel('Vector length')

    ax[1].set_ylabel('H1 / log2(length)')
    ax[1].legend()
    plt.tight_layout()
    return fig

@load_or_save_fig('app/assets_produced/3_Diffusion_theory/compare_noise_metrics.png', deactivate=deactivate)
def compare_noise_metrics():
    T = 501
    vs = VarianceSchedule(T, method="cosine", epsilon=0.08)
    img_path = 'app/assets/example_images/dog.png'
    img = prep_image(img_path, size=(512, 512   ))
    noise_type='normal'
    data = {t:{'img': vs(img, t, clip=False, noise_type=noise_type)} for t in range(T)}

    for t, im in data.items():
        im_arr = im['img'].clone().detach()
        H1 = shannon_entropy_binned(im_arr.flatten())
        # st.write('H1:', H1)
        H2 = shannon_entropy_2d(im_arr)
        # st.write('H2:', H2)
        KL = kl_score(im_arr)
        # st.write('KL:', KL)
        data[t]['H1'] = H1
        data[t]['H2'] = H2
        data[t]['KL'] = KL


    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 6, figure=fig)
    ax_ims = [plt.subplot(gs[0, i]) for i in range(6)]
    ax_scores = [plt.subplot(gs[1, i:i+2]) for i in range(0,6,2)]

    t_of_ims_to_show = np.linspace(0, T-1, 6, dtype=int)
    for t, ax in zip(t_of_ims_to_show, ax_ims):
        ax.imshow(data[t]['img'], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t={t}")

    H1 = [data[t]['H1'] for t in range(T)]
    H2 = [data[t]['H2'] for t in range(T)]
    KL = [data[t]['KL'] for t in range(T)]

    n_pure_noise = 100
    H1_pure_noise = [shannon_entropy_binned(torch.randn_like(img).flatten())
                        for _ in range(n_pure_noise)]
    H1_pure_noise = np.mean(H1_pure_noise), np.std(H1_pure_noise)

    H2_pure_noise = [shannon_entropy_2d(torch.randn_like(img), return_all=False)
                        for _ in range(n_pure_noise)]
    H2_pure_noise = np.mean(H2_pure_noise), np.std(H2_pure_noise)

    KL_pure_noise = [kl_score(torch.randn_like(img))
                        for _ in range(n_pure_noise)]
    KL_pure_noise = np.mean(KL_pure_noise), np.std(KL_pure_noise)

    # ax 0: H1
    ax_scores[0].plot(H1, label='Sample', color='purple')
    ax_scores[0].axhline(H1_pure_noise[0], color='orangered', label='Pure noise')
    ax_scores[0].fill_between([0, T], H1_pure_noise[0] - H1_pure_noise[1], H1_pure_noise[0] + H1_pure_noise[1],
                            color='orangered', alpha=0.3)
    ax_scores[0].set_title('Shannon entropy')
    ax_scores[0].set_xlabel('t')
    ax_scores[0].set_ylabel('H1')
    # ax_scores[0].legend()

    # ax 1: H2
    ax_scores[1].plot(H2, label='Sample', color='purple')
    ax_scores[1].axhline(H2_pure_noise[0], color='orangered', label='Pure noise')
    ax_scores[1].fill_between([0, T], H2_pure_noise[0] - H2_pure_noise[1], H2_pure_noise[0] + H2_pure_noise[1],
                            color='orangered', alpha=0.3)
    ax_scores[1].set_title('Shannon entropy (2d)')
    ax_scores[1].set_xlabel('t')
    ax_scores[1].set_ylabel('H2')
    # ax_scores[1].legend(ncol=2)


    # ax 2: KL
    ax_scores[2].plot(KL, label='Sample', color='purple')
    ax_scores[2].axhline(KL_pure_noise[0], color='orangered', label='Pure noise')
    ax_scores[2].fill_between([0, T], KL_pure_noise[0] - KL_pure_noise[1], KL_pure_noise[0] + KL_pure_noise[1],
                            color='orangered', alpha=0.3)

    ax_scores[2].set_title('KL divergence from $\mathcal{N}(0,1)$')
    ax_scores[2].set_xlabel('t')
    ax_scores[2].set_ylabel('KL')
    ax_scores[2].legend()


    plt.tight_layout()
    return fig

def measure_noise_level_page():
    # Introduction
    st.write("""    
    To gain information, in reagards to the level of noise in a sample, we explore the use of three metrics: Shannon entropy, its extension to 2d, and the Kullback-Leibler (KL) divergence.
    """)

    # Shannon 1d entropy
    st.write('#### Shannon entropy')
    cols = st.columns(2)

    with cols[0]:
        st.write(r"""
        
        For a vector in n-dimensional space, we consider the distribution of its component-values. We bin the values into $\sqrt{n}$ bins, as suggested in \cite{https://arxiv.org/pdf/1609.01117.pdf}, where $n$ is the length of the vector. We then calculate the Shannon entropy of the distribution of the values in the bins as follows:
                
        $$
        H = -\sum p(x) \log_2 p(x)
        $$


        So, a vector with with its magnitude spread across its components has maximum entropy, while a vector with all its magnitude concentrated in a single component has minimum entropy.
                
        A problem with Shannon entropy, is the dependence on vector scale. To make the entropy scale-invariant, we can use the following fit a log function to the entropy as a function of the vector length.
                
        We could naturally consider the Shannon entropy a relevant measure for data of any shape, as we could just flatten the tensor, but we find a slightly modified measure for image-shaped samples [17]).
        """)  

    with cols[1]:
        fig = shannon_entropy_scale_dependence_plot()
        st.pyplot(fig)
        st.caption("Top: Shannon entropy of Gaussian noise vectors of different lengths. Bottom: Shannon entropy normalized by the log of the length of the vector.")
    st.divider()
    st.write('#### 2d Shannon entropy')
    st.write(r"""
            \cite{larkin2016reflections} suggest taking the partial spatial derivative along the width and height of an image, and summing them. Then, to flatten the array and perform the typical 1d Shannon entropy measure.
                
            $$
                H_2(\nabla f)  = -\sum_{j=1}^J\sum_{i=1}^I p_{i,j} \log_2(p_{i,j})
            $$""" )
    st.divider()
    st.write('#### KL score')
    st.write(r"""
            Since our sample tends towards the noise distribution as $t\rightarrow\infty$, we can measure the difference between our sample distribution. The Kullback–Leibler divergence, describes the difference between two distributions, $P$ and $Q$; and is defined by the expected difference between the distributions:
            $$
                \mathcal{D}_\text{KL}(P||Q) 
                = 
                \sum_{x\in X} P(x)\log\frac{P(x)}{Q(x)}
                =
                \braket{\log P(x) - \log Q(x)}.
            $$""")

    st.divider()
    st.write("""## Comparing noise metrics""")

    fig = compare_noise_metrics()
    st.pyplot(fig)
    st.caption("Top row: images with added noise. Bottom row: metrics for measuring noise level. Red line: mean of 10 samples of pure noise, with the shaded area demarking oone standard deviation.")

    st.write("""
        Notice, the Shannon 1d entropy is better than the 2d entropy for measuring noise level. This is clear as the 2d entropy, almost instantly says there is the level of noise we see in a pure noise sample.

        The best measure of the noise level is the KL score, which is the KL divergence between the input tensor and the standard normal distribution.
    """)
