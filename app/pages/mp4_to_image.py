import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# for loading video: cv2
import cv2
path = 'assets/recon_true.mp4'

st.write("## Video to Image")

cols = st.columns(2)

with cols[0]:
    st.video(path)

with cols[1]:

    def prep_frame(frame):
        frame = frame.astype(float)
        frame/=255.
        frame += 0
        frame = frame[100:,:] # crop the top


        frame[frame > .9] = 0.0
        # k = st.slider("Quantization", 1, 256, 3)
        # frame = (frame/k).astype(int)*k
        return frame

    video = cv2.VideoCapture(path)
    ret, frame = video.read()
    video.release()
    frame = prep_frame(frame)
    frame.shape
    fig, ax = plt.subplots()
    ax.imshow(frame, cmap='gray')
    ax.set_title("Frame")
    ax.axis('off')

    st.pyplot(fig)
    # st.image()
    # frame
    fig, ax = plt.subplots()
    ax.hist(np.linalg.norm(frame, axis=2).flatten(), fc='k', ec='k')
    ax.set_title("Histogram of frame")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    ax.set_yscale('log')
    st.pyplot(fig)
    
    video = cv2.VideoCapture(path)
    frames = []
    for i in range(10):
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)

    frames = np.array(frames)
    # frames
    frames[frames > 250] = 0
    1, frames.shape    
    

    st.image(np.mean(frames, axis=0), clamp=True)
    video.release()



st.divider()

'Hmm, this is not working, lets do it from points'
