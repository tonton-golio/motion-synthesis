import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# logging utils (plotting)
def plot_3d_pose(data, index, ax=None, **kwargs):
    """
    Plot a single 3D pose.

    Args:
    - data (np.array): 3D pose data (seq_len, joints_num, 3)
    - index (int): index of the frame to plot
    """

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

    kinematic_chain = [
        [0, 2, 5, 8, 11],
        [0, 1, 4, 7, 10],
        [0, 3, 6, 9, 12, 15],
        [9, 14, 17, 19, 21],
        [9, 13, 16, 18, 20],
    ]

    colors = [
        "purple",
        "purple",
        "black",
        "purple",
        "purple",
    ]

    for i, (chain, color) in enumerate(zip(kinematic_chain, colors)):
        ax.plot3D(
            data[index, chain, 0],
            data[index, chain, 1],
            data[index, chain, 2],
            linewidth=4.0 if i < 5 else 2.0,
            color=color,
            **kwargs
        )
        # plot the joints
        ax.scatter(
            data[index, chain, 0],
            data[index, chain, 1],
            data[index, chain, 2],
            color=color,
            s=3,
            **kwargs
        )

    plt.axis("off")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return ax

def plot_xzPlane(ax, minx, maxx, minz, miny, maxy):
    ## Plot a plane XZ
    verts = [
        [minx, miny, minz, ],
        [minx, maxy, minz, ],
        [maxx, maxy, minz, ],
        [maxx, miny, minz, ],
    ]
    xz_plane = Poly3DCollection([verts])
    xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
    ax.add_collection3d(xz_plane)

def init_3d_plot(ax, fig, title, radius=2, **kwargs):
    ax.set_xlim3d([kwargs.get("minx", -radius), 
                   kwargs.get("maxx", radius)])
    ax.set_ylim3d([kwargs.get("miny", 0),
                    kwargs.get("maxy", radius)])
    ax.set_zlim3d([kwargs.get("minz", -radius), 
                   kwargs.get("maxz", radius)])

    ax.view_init(elev=kwargs.get("elev", 110),
                 azim=kwargs.get("azim", -90))
    # print(title)
    title_sp = title.split(" ")
    if len(title_sp) > 10:
        title = "\n".join([" ".join(title_sp[:10]), " ".join(title_sp[10:])])
    fig.suptitle(title, fontsize=20)
    # ax.grid(b=False)

def plot_trajec(trajec, index, ax):
    ax.plot3D(
        trajec[:index, 0] - trajec[index, 0],
        np.zeros_like(trajec[:index, 0]),
        trajec[:index, 1] - trajec[index, 1],
        linewidth=1.0,
        color="blue",
    )


cols = st.columns(2)
with cols[0]:
    # selector
    mirrored = st.checkbox("Mirrored", False)
    vnum = st.number_input("Version", 1, 60, 1)
    v = '0'* (6-len(str(vnum))) + str(vnum)
    v = v if not mirrored else f'M{v}'
    # load data
    text_path = f'../stranger_repos/HumanML3D/HumanML3D/texts/{v}.txt'
    joints_path = f'../stranger_repos/HumanML3D/HumanML3D/new_joints/{v}.npy'
    with open(text_path, 'r') as f:
        text = f.read()
        text = text.split('#')[0]
    
    x = np.load(joints_path)
    x.shape

    # swap x,z,y to z,x,y
    x = x[:, :, [0, 2,1]]

    num_images = st.slider("Number of images", 1, 30, 30)


with cols[1]:
    st.write('**Camera settings**')
    subcols = st.columns(2)
    with subcols[0]:
        elev = st.slider("Elevation", -180, 180, 0)
        min_x = st.slider("Min X", -2., 0., -1.)
        min_y = st.slider("Min Y", 0., 2., 0.)
        min_z = 0#st. slider("Min Z", -2., 0., -1.)
    with subcols[1]:
        azim = st.slider("Azimuth", -180, 180, -0)
        max_x = st.slider("Max X", 0., 2., 1.)
        max_y = st.slider("Max Y", 0., 4., 1.)
        max_z = 1.2#st.slider("Max Z", 0., 2., 1.)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(20, 8))
init_3d_plot(ax, fig, text, elev=elev, azim=azim, miny=min_y, maxy=max_y, minx=min_x, maxx=max_x, minz=min_z, maxz=max_z)
min_ = np.min(x.reshape(-1,3), axis=0)
max_ = np.max(x.reshape(-1,3), axis=0)
# plot_xzPlane(ax, -1, 1, 0, 0, 2)
plot_xzPlane(ax, min_[0], max_[0], 0, min_[1], max_[1])
plot_trajec(x[:, 0, :2], 0, ax)

alphas = np.linspace(0.5, 1, x.shape[0])

frames_to_plot = np.linspace(0, x.shape[0], num_images, endpoint=False, dtype=int)
alphas = alphas[frames_to_plot]
alphas/=alphas.max()
alphas[alphas<1] = (alphas[alphas<1]+0.5)/2
alphas **=2



# alphas = [0.5, 0.5, 0.5, 1]
# frames_to_plot = [0,15, 30, 60, 159]
# alphas = (len(frames_to_plot)-1)*[0.5] + [1]


for i, alpha in zip(frames_to_plot, alphas):
    
    plot_3d_pose(x, i, ax, alpha=alpha)

plt.tight_layout()

st.pyplot(fig)