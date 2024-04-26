import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
vl = vector_length = 10

def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0: 
       return v
    return v / norm

A = normalize(np.random.rand(vl))
B = normalize(np.random.rand(vl)**99)


"""
How is the length of a vector split across its components?

If we have greater distribution of values, this is greater entropy.
"""

def shannon_entropy_1d(v):
    v = normalize(v)
    return -np.sum(v * np.log(v))

print(shannon_entropy_1d(A), shannon_entropy_1d(B))

fig, ax = plt.subplots(1, 2, figsize=(10, 2))
# ax[0].bar(range(vl), A)
# ax[1].bar(range(vl), B)
ax[0].imshow(A.reshape(-1, 1), aspect='auto', vmin=0, vmax=1, cmap='RdBu_r')
rightplot = ax[1].imshow(B.reshape(-1, 1), aspect='auto', vmin=0, vmax=1, cmap='RdBu_r')
plt.colorbar(rightplot, ax=ax[1])
ax[0].set_title(f'Distributed vector $a$ with entropy H={shannon_entropy_1d(A):.2f}')
ax[1].set_title(f'Distributed vector $b$ with entropy H={shannon_entropy_1d(B):.2f}')

for a in ax:
    a.set_xticks([])
    # a.set_yticks([])
ax[0].set_ylabel('Vector component')

st.pyplot(fig)