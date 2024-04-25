import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time

# set page layout to wide
st.set_page_config(layout="wide")

def make_df():
    file = 'autoEncoder/assets/generated_motion/transformer/E9_-_side_step_left_stageii20231207-111812.npz'
    #file = 'data/extracted_joint_positions/B7_-_walk_backwards_turn_forwards_stageii.npz'
    seq = np.load(file, allow_pickle=True)['arr_0'].squeeze()
    print(seq.shape)
    df_gen = pd.DataFrame(seq.reshape(-1, 3), columns=['x', 'y', 'z'])

    frame_number = np.array([[i] * len(seq[i]) for i in range(len(seq))]).flatten()
    joint_number = np.array([np.arange(len(seq[i])) for i in range(len(seq))]).flatten()
    df_gen['frame_number'] = frame_number
    df_gen['joint_number'] = joint_number

    return df_gen



def make_plotly(df_gen, frame_number, joint_connections):

    fig = go.Figure()

    df_frame = df_gen[df_gen['frame_number'] == frame_number]

    # plot scatter
    fig.add_trace(go.Scatter3d(x=df_frame['x'], y=df_frame['y'], z=df_frame['z'], mode='markers', name='Joints', hovertext=df_frame['joint_number']))

    for i, (start,stop) in enumerate(joint_connections):
        if start == None or stop == None: continue
        x0, y0, z0 = df_frame[df_frame['joint_number'] == start][['x', 'y', 'z']].values[0]
        x1, y1, z1 = df_frame[df_frame['joint_number'] == stop][['x', 'y', 'z']].values[0]

        fig.add_trace(go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1], mode='lines', name=f'Joint {i}', line=dict(color='red', width=4)))

    # hide legend
    fig.update_layout(showlegend=False)

    # set axis range
    # fig.update_layout(
    #     scene=dict(
    #         xaxis=dict(range=[-3, 3],),
    #         yaxis=dict(range=[-3, 3],),
    #         zaxis=dict(range=[-3, 3],),
    #     ),
    # )

    # set height to 500
    fig.update_layout(height=1000)

    return fig



df_gen = make_df()


cols = st.columns((1,4))

with cols[0]:
    st.write('Frame number')
    frame_number = st.slider('Frame number', 0, len(df_gen['frame_number'].unique()) - 1, 0)
    
    st.write('joint_connections')

    joint_connections = [
        (0,3),
        (1,0),
        (2,0),
        (3,6),
        (4,1),
        (5,2),
        (6,9),
        (7,4),
        (8,5),
        (9,12),
        (10,27),
        (11,30),
        (12, 12),
        (13,9),
        (14,9),
        (15,12),
        (16,13),
        (17,14),
        (18,16),
        (19,17),
        (20,18),
        (21,19),
        (22,12),
        (23,22),
        (24,15),
        (25,26),
        (26,10),
        (27,7),
        (28,29),
        (29,11),
        (30,8),
    ]

    # # let the user connect each joins to 1 other joint
    # conn = [st.select_slider(f'Joint {0}', [None]+list(np.arange(31)), 3),
    #         st.select_slider(f'Joint {1}', [None]+list(np.arange(31)), 0),
    #         st.select_slider(f'Joint {2}', [None]+list(np.arange(31)), 0),
    #         st.select_slider(f'Joint {3}', [None]+list(np.arange(31)), 6),
    #         st.select_slider(f'Joint {4}', [None]+list(np.arange(31)), 1),
    #         st.select_slider(f'Joint {5}', [None]+list(np.arange(31)), 2),
    #         st.select_slider(f'Joint {6}', [None]+list(np.arange(31)), 9),
    #         st.select_slider(f'Joint {7}', [None]+list(np.arange(31)), 4),
    #         st.select_slider(f'Joint {8}', [None]+list(np.arange(31)), 5),
    #         st.select_slider(f'Joint {9}', [None]+list(np.arange(31)), 12),

    #         st.select_slider(f'Joint {10}', [None]+list(np.arange(31)), 27),
    #         st.select_slider(f'Joint {11}', [None]+list(np.arange(31)), 30),
    #         st.select_slider(f'Joint {12}', [None]+list(np.arange(31)), ),
    #         st.select_slider(f'Joint {13}', [None]+list(np.arange(31)), 9),
    #         st.select_slider(f'Joint {14}', [None]+list(np.arange(31)), 9),
    #         st.select_slider(f'Joint {15}', [None]+list(np.arange(31)), 12),
    #         st.select_slider(f'Joint {16}', [None]+list(np.arange(31)), 13),
    #         st.select_slider(f'Joint {17}', [None]+list(np.arange(31)), 14),
    #         st.select_slider(f'Joint {18}', [None]+list(np.arange(31)), 16),
    #         st.select_slider(f'Joint {19}', [None]+list(np.arange(31)), 17),
    #         st.select_slider(f'Joint {20}', [None]+list(np.arange(31)), 18),

    #         st.select_slider(f'Joint {21}', [None]+list(np.arange(31)), 19),
    #         st.select_slider(f'Joint {22}', [None]+list(np.arange(31)), 12),
    #         st.select_slider(f'Joint {23}', [None]+list(np.arange(31)), 22),
    #         st.select_slider(f'Joint {24}', [None]+list(np.arange(31)), 15),
    #         st.select_slider(f'Joint {25}', [None]+list(np.arange(31)), 26),
    #         st.select_slider(f'Joint {26}', [None]+list(np.arange(31)), 10),
    #         st.select_slider(f'Joint {27}', [None]+list(np.arange(31)), 7),
    #         st.select_slider(f'Joint {28}', [None]+list(np.arange(31)), 29),
    #         st.select_slider(f'Joint {29}', [None]+list(np.arange(31)), 11),
    #         st.select_slider(f'Joint {30}', [None]+list(np.arange(31)), 8),


            
    #         ]

    # joint_connections = [(i, j) for i, j in enumerate(conn)]
    # joint_connections

    run = st.button('Run')
    stop = st.button('Stop')

with cols[1]:

    canvas = st.empty()
    if run:
        
        for frame_number in range(len(df_gen['frame_number'].unique())):

            canvas.plotly_chart(make_plotly(df_gen, frame_number, joint_connections), use_container_width=True)
            
            time.sleep(0.1)
            if stop: 
                break
                run = False
        run = False
    else:
        canvas.plotly_chart(make_plotly(df_gen, frame_number, joint_connections), use_container_width=True)