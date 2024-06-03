import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Title and intro
"""
# Autoencoding Theory

"""

tab_names = [
    'How much code?',
    'Make GIF',
    'twitch chat',
    'storage usage',
    'PCA demo',
    'hparams viewer',
    'Determine joint connectivity',
    'Download data',
    'links for later'
]
tabs = {name: tab for name, tab in zip(tab_names, st.tabs(tab_names))}


with tabs['How much code?']:
    import os

    # List of folders to search
    folders = ["../app", "../mnist_latent_diffusion", 
               "../motion_latent_diffusion"]
    
    exclude = 'logs'

    def count_in_file(file_path):
        with open(file_path, 'r') as file:
            f = file.read()

        return dict(
            lines=f.count('\n'),
            words=len(f.split()),
            chars=len(f)
        )

    def count_lines_in_folder(folder):
        total = {'lines': 0, 'words': 0, 'chars': 0}
        for root, _, files in os.walk(folder):
            
            if exclude in root:
                continue
            
            for file in files:
                if file.endswith('.py'):
                    # st.write(os.path.join(root, file))
                    file_path = os.path.join(root, file)
                    counts = count_in_file(file_path)
                    total['lines'] += counts['lines']
                    total['words'] += counts['words']
                    total['chars'] += counts['chars']
        return total

    def main():
        data = {}
        for folder in folders:
            if os.path.exists(folder):
                count = count_lines_in_folder(folder)
                # st.write(f"Folder: {folder}, Lines: {count['lines']}, Words: {count['words']}, Chars: {count['chars']}")
                data[folder.split('../')[1]] = count
            else:
                print(f"Folder not found: {folder}")
        df = pd.DataFrame(data)
        df['Total'] = df.sum(axis=1)
        df

    if __name__ == "__main__":
        main()

# Make GIF
with tabs['Make GIF']:
    import streamlit as st
    import os
    import glob
    import subprocess

    # Title and introduction
    """
    ### Make GIF

    Supply a folder, and a output destination, and this script will make a gif from the images in the folder.
    """

    def get_file_names(path):
        return sorted(glob.glob(path + "/*.png"))

    # Write file names to a text file for ffmpeg
    def write_to_file(file_names, output_file):
        with open(output_file, 'w') as file:
            for name in file_names:
                file.write(f"file '{name}'\n")
                file.write(f"duration 0.5\n")  # Adjust duration as needed

    # Create a gif using ffmpeg
    def create_gif(input_path, output_path):
        file_names = get_file_names(input_path)
        temp_file = "file_list.txt"
        for f in file_names:
            print(f)    
        write_to_file(file_names, temp_file)
        command = f'ffmpeg -f concat -safe 0 -i {temp_file} -vf "fps=30,scale=168:168:flags=lanczos" -c:v gif {output_path} -y'
        subprocess.run(command, shell=True)
        os.remove(temp_file)  # Clean up the temporary file

    im_folder = st.text_input("Enter the folder path")
    out_name = st.text_input("Enter the output name")
    if st.button("Make gif"):
        create_gif(im_folder, out_name)
        st.write(f"Created gif at {out_name}")


# twitch chat connector
with tabs['twitch chat']:
    st.code("""
            from twitch_chat_irc import twitch_chat_irc as tci
            connection = tci.TwitchChatIRC()
            connection.listen('iamtontonton')
            """)
    st.write("This will connect to the twitch chat of 'iamtontonton'. If you are missing the module:")
    st.code("""
    !pip install twitch-chat-irc
    """)


# storage usage
with tabs['storage usage']:
    # determine how much storage is used by a given directory
    # we want the results in a dataframe
    import streamlit as st
    import os, sys, glob
    import pandas as pd

    def get_file_sizes(folder_path):
        data = {}
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)
                data[file_path] = size

        df = pd.DataFrame(list(data.items()), columns=['file', 'size'])
        return df

    def expand_info(df):
        #df['size_kb'] = df['size'] / 1024
        df['size_mb'] = df['size'] / 1024 / 1024
        df['size_gb'] = df['size_mb'] / 1024
        
        df['file_extension'] = df['file'].apply(lambda x: x.split('.')[-1] if not 'ubyte' in x else 'ubyte')
        df['parent_folder'] = df['file'].apply(lambda x: x.split('/')[-2])
        df['file_name'] = df['file'].apply(lambda x: x.split('/')[-1])
        df['grandparent_folder'] = df['file'].apply(lambda x: x.split('/')[-3] if len(x.split('/')) > 2 else None)
        df['great_grandparent_folder'] = df['file'].apply(lambda x: x.split('/')[-4] if len(x.split('/')) > 3 else None)

    def accumulate_folder_sizes(folder_path):
        folder_sizes = {}
        
        # Function to recursively accumulate sizes
        def accumulate(path, container):
            if os.path.isdir(path):
                total_size = 0
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    item_size = accumulate(item_path, container.setdefault(item, {}))
                    total_size += item_size
                container["__size__"] = total_size  # Store the total size in the current folder's dict
                return total_size
            else:
                return os.path.getsize(path)
        
        # Start the accumulation
        accumulate(folder_path, folder_sizes)
        return folder_sizes

    def print_tree(container, indent=""):
        for key, value in container.items():
            if key == "__size__":
                continue  # Skip printing the size key directly
            if isinstance(value, dict):
                print(f"{indent}{key}/ ({value.get('__size__', 0)} bytes)")
                print_tree(value, indent + "  ")
            else:
                pass#print(f"{indent}{key}: {value} bytes")

    if __name__ == "__main__":

        # set up
        folder_path = '../'
        df = get_file_sizes(folder_path)
        expand_info(df)
        df = df.sort_values('size', ascending=False)  # sort by size
        total_size = df['size_gb'].sum()


        # render
        print(f'Total size: {total_size:.2f} GB')

        # total size by file extension
        print((df.groupby('file_extension')['size_gb'].sum().sort_values(ascending=False).astype(int).astype(str) + ' GB')[:7])
        count_by_ext = df.groupby('file_extension').count()['size'].sort_values(ascending=False)
        count_by_ext
        # total size by parent folder
        print((df.groupby('parent_folder')['size_gb'].sum().sort_values(ascending=False).astype(int).astype(str) + ' GB')[:7])

        # total size by grandparent folder
        print((df.groupby('grandparent_folder')['size_gb'].sum().sort_values(ascending=False).astype(int).astype(str) + ' GB')[:7])

        # total size by great grandparent folder
        print((df.groupby('great_grandparent_folder')['size_gb'].sum().sort_values(ascending=False).astype(int).astype(str) + ' GB')[:7])

        """# Storage usage of repository"""
        st.metric(label="Total size", value=f"{total_size:.2f} GB")
        cols = st.columns(2)
        with cols[0]: # total size by file extension
            st.dataframe((df.groupby('file_extension')['size_gb'].sum().sort_values(ascending=False).astype(int).astype(str) + ' GB')[:5])
        with cols[1]: # total size by parent folder
            st.dataframe((df.groupby('parent_folder')['size_gb'].sum().sort_values(ascending=False).astype(int).astype(str) + ' GB')[:5])

        with cols[0]: # total size by grandparent folder
            st.dataframe((df.groupby('grandparent_folder')['size_gb'].sum().sort_values(ascending=False).astype(int).astype(str) + ' GB')[:5])

        with cols[1]: # total size by great grandparent folder
            st.dataframe((df.groupby('great_grandparent_folder')['size_gb'].sum().sort_values(ascending=False).astype(int).astype(str) + ' GB')[:5])


        # with st.expander("More info`"):
        #     folder_sizes = accumulate_folder_sizes(folder_path)
        #     folder_sizes


# PCA demo
with tabs['PCA demo']:
    import streamlit as st
    # PCA explanation

    import numpy as np
    import matplotlib.pyplot as plt

    '### Principal Component Analysis (PCA)'

    def PCA_demo(n_samples=100, plot=True):
        # Generate data
        # np.random.seed(1)

        # Generate 2D data
        mu_vec = np.array([0, .5])
        cov_matrix = np.array([[.7, -3], [0, 1]]) 
        x = np.random.multivariate_normal(mu_vec, cov_matrix, n_samples).T

        # Compute covariance matrix and eigen values
        cov_measured = np.cov(x)  # measure covariance
        eig_vals, eig_vecs = np.linalg.eig(cov_measured)  # diagonalize

        # sort eigen values in descending order
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        # show eigenvectors
        
        if plot:
            mean_x1 = np.mean(x, axis=1)  # compute mean vector for plotting

            l = length_factor = 6  # length factors, to make plot nicer
            l2 = length_factor /2

            fig = plt.figure(figsize=(6.5, 6.5))
            plt.scatter(x[0], x[1], marker='x', color='red', alpha=0.4, label='samples')

            plt.plot([mean_x1[0]-l*eig_pairs[0][1][0], mean_x1[0] + l * eig_pairs[0][1][0]], 
                    [mean_x1[1]-l*eig_pairs[0][1][1], mean_x1[1] + l * eig_pairs[0][1][1]],
                        color='black', linewidth=3, label='Principal Axis 1')  
            plt.plot([mean_x1[0]-l*eig_pairs[1][1][0], mean_x1[0] + l * eig_pairs[1][1][0]],
                        [mean_x1[1]-l2*eig_pairs[1][1][1], mean_x1[1] + l2 * eig_pairs[1][1][1]],
                            color='darkgrey', linewidth=3, label='Principal Axis 2')  

            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            plt.xlim([-5, 5])
            plt.ylim([-5, 5])
            plt.legend(loc='lower left')

            # plt.savefig('../assets/PCA_demonstration.png')
            return fig
        
        return eig_pairs


    cols = st.columns((2, 2))
    fig = PCA_demo(n_samples=420)
    cols[0].write("""
                PCA is a technique for linear dimensionality reduction. It can be explained in two ways; 1. more intuitive, 2. more mathematical.

                1. *Intuitive explanation*: We place the first principal axis, $\\text{A}_1$, along the direction in our data cloud, which contains the greatest variance (see the black line on the plot). Then in the space orthogonal to $\\text{A}_1$, we repeat the process, see the grey line.

                2. *Mathematical explanation*: Principal axes are the eigenvectors of the covariance matrix of the data. The eigenvalues of the covariance matrix represent the variance along the principal axes.
                
                We define some data $X in \mathbb{R}^{n \times m}$, where $n$ is the number of samples and $m$ is the number of features. We compute the covariance between features
                    $$
                \\text{cov}(X) = \mathbb{E}[(X - \mathbb{E}[X])(X - \mathbb{E}[X])^T].
                $$
                The eigenvectors of $C$ are the principal axes, and the eigenvalues are the variance along these axes.

                """)

    cols[1].pyplot(fig)


# hparams viewer
# with tabs['hparams viewer']:
#     import streamlit as st
#     import os
#     from glob import glob
#     import yaml

#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     with st.sidebar:
#         log_dirs = ['logs/mnistVAEoptuna/', 'MNIST_VAE', 'MNIST_Diffusion', 'CIFAR10_Diffusion']
#         log_dir = st.selectbox('Select log directory', log_dirs)



#     def make_df(path_outer = 'logs/mnistVAEoptuna/'):

#         log_dirs = glob('../mnist_latent_diffusion/'+path_outer + '*/')

#         data = {}
#         for dir in log_dirs:
#             num = dir.split('/')[-2].split('_')[-1]
#             print(dir)
#             # check if hparams.yaml exists
#             if not os.path.exists(dir + 'hparams.yaml'):
#                 print('hparams.yaml not found')
#                 continue

#             with open(dir + 'hparams.yaml') as file:
#                 hparams = yaml.load(file, Loader=yaml.FullLoader)

#             data[num] = hparams

#         df = pd.DataFrame(data).T
#         df.dropna(inplace=True)

#         return df

#     df = make_df(log_dir)
#     df

#     # fig 1
#     plt.style.use('seaborn-v0_8-paper')
#     fig = plt.figure(figsize=(10, 6))
#     for ld in sorted(df['latent_dim'].unique()):
#         df_ld = df[df['latent_dim'] == ld]
#         kl = df_ld['klDivWeight']
#         mse = df_ld['mse_us_tst']
#         # print(f'Latent dim: {ld}')
#         # print('KL Div Weight:', kl)
#         # print('MSE:', mse)
#         # print()
#         plt.scatter(df_ld['klDivWeight'], df_ld['mse_us_tst'], label=str(ld))

#     plt.legend(title='Latent dim')
#     plt.xlabel('Kullback-Leibler divergence weight')
#     plt.ylabel('MSE (unscaled) on test set')
#     plt.xscale('log')
#     # plt.yscale('log')
#     # plt.grid()
#     # plt.xlim( 1e-6, 1e-4,)
#     # plt.savefig('assets/mnistVAEoptuna_klDivWeight_vs_mse_us_tst.png')
#     st.pyplot(fig)


#     # fig 2
#     fig = plt.figure(figsize=(10, 6))
#     sns.heatmap(df.corr())
#     st.pyplot(fig)



#     # fig 3: clustering
#     import umap
#     import matplotlib.pyplot as plt
#     from sklearn.cluster import KMeans



#     df.dropna(inplace=True)
#     X = df.values
#     y = df['latent_dim'].values
#     mse = df['mse_us_tst'].values


#     reducer = umap.UMAP()
#     embedding = reducer.fit_transform(X)

#     # cluster
#     kmeans = KMeans(n_clusters=3, random_state=0).fit(embedding)
#     fig = plt.figure(figsize=(10, 6))
#     plt.scatter(embedding[:, 0], embedding[:, 1], c=mse, cmap='viridis', s=10)
#     plt.gca().set_aspect('equal', 'datalim')
#     plt.colorbar()


#     plt.legend()
#     st.pyplot(fig)



#     """
#     import os, sys, glob



#     import pandas as pd
#     import yaml
#     import matplotlib.pyplot as plt
#     import numpy as np

#     def get_data(folders):
#         data = {}
#         for folder in folders:
#             folder_num = int(folder.split('/')[-1].split('_')[-1])
#             with open(folder + '/hparams.yaml', 'r') as f:
#                 params = yaml.load(f, Loader=yaml.FullLoader)
#                 data[folder_num] = params

#         df = pd.DataFrame(data).T
#         df.sort_index(inplace=True)
#         return df

#     def map_categoricals(df, col_name):
#         mapper = {k: i for i, k in enumerate(df[col_name].unique())}
#         df[col_name+'_mapped'] = df[col_name].map(mapper)
#         return df, mapper

#     def sort_by_mean(df, x_col, target_col):
#         mean = df.groupby(x_col).mean()

#         # sort by mean
#         category_names = mean[target_col].sort_values().index
#         # print(category_names)
#         #sort df by category_names
#         # print(df)
#         df = df.set_index(x_col).loc[category_names].reset_index()
#         # print(df)
#         mapper = {k: i for i, k in enumerate(df[x_col].unique())}
#         df[x_col+'_mapped'] = df[x_col].map(mapper)
#         return df

#     def hyper_param_chart(df, x_col, target_col, ax, normalize_y=True):
#         # we normalize all columns, and plot lines for each row
#         # scale to between 0 and 1
        

#         y = df[target_col].values.flatten()
#         x = df[x_col+'_mapped'].values.flatten()
#         try:
#             x = (x - x.min()) / (x.max() - x.min())
#         except:
#             print('x scaled failed')
#             x = x
        
#         try:
#             y_scaled = (y - y.min()) / (y.max() - y.min())
#         except:
#             print('y scaled failed')
#             y_scaled = y
#         if normalize_y:
#             y = y_scaled
#         # print('x', x)
#         # print('y', y)
        
#         ax2 = ax.twinx()
#         ax.set_ylim(-.1, 1.1)
#         ax2.set_ylim(-.1, 1.1)
#         # print(mapper)
#         # ax.set_yticks(np.linspace(0, 1, len(mapper)), mapper.keys())

#         # color by y value
#         for i in range(len(x)):
#             ax.plot([0, 1], [x[i], y[i]], color=plt.cm.RdBu(y_scaled[i]))
#             # ax.plot(range(2), [x[i], y[i]], label=i)

#         ax.grid()

#     def hyperparam_error_bar_chart(data, ax,  x_col='ACTIVATION',  target_col='mse_unscaled_test'):
#         # print(data)
#         # data = sort_by_mean(data, x_col, target_col)
#         mean = data.groupby(x_col).mean()
#         std = data.groupby(x_col).std()
#         try:
#             mean_scaled = ((mean - mean.min()) / (mean.max() - mean.min()))[target_col]
#         except:
#             print('mean scaled failed')
#             mean_scaled = mean[target_col]

#         # sort by mean
#         category_names = mean[target_col].sort_values().index
#         mean = mean.loc[category_names]
#         std = std.loc[category_names]

#         # print(mean)
#         # print(std)
#         for i, (m, s) in enumerate(zip(mean[target_col], std[target_col])):
#             ax.errorbar(m, i, xerr=s, fmt='o', color=plt.cm.viridis(mean_scaled[i]))

#         ax.set_yticks(range(len(mean.index)), mean.index)
#         # return mean, std

#     # count how many times each activation function yields a binned mse
#     def bin_plot(data, bins, ax,  x_col='ACTIVATION', target_col='mse_unscaled_test'):
#         # data = sort_by_mean(data, x_col, target_col)

#         bins = np.linspace(data[target_col].min(), data[target_col].max(), bins)
#         bin_labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
#         counts = {}
#         for i in range(len(data)):
            
#             act = data[x_col].iloc[i]
#             val = data[target_col].iloc[i]
#             # print(act, val)
#             if act not in counts:
#                 counts[act] = np.zeros(len(bins)-1)
#             for j in range(len(bins)-1):
#                 if val >= bins[j] and val < bins[j+1]:
#                     counts[act][j] += 1
#                     break

        
        
#         # print(counts)
#         # Ensure every activation has a count for each bin, even if it's 0
#         all_bins = np.zeros(len(bins)-1)
#         for act in counts:
#             counts[act] = counts[act] + all_bins
#         colors = plt.cm.RdBu(np.linspace(0,1, len(bins)))
#         # Prepare data for stacked bar chart
        
#         for i, (act, vals) in enumerate(counts.items()):
#             # print(act)
#             bottom = 0
            
#             for c, v in zip(colors, vals):
#                 ax.barh(i+bottom-.25, v, height=.1, color=c)
#                 bottom += .1

            

#         # show the bins in a legend
#         labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
#         handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(bins)-1)]
#         ax.legend(handles, labels)
#         ax.set_xlim(-1, 12)

#         ax.set_yticks(range(len(counts)), counts.keys())


#     if __name__ == '__main__':
#         path = 'tb_logs_activation_sweep_no_prune_100_epochs/MNISTAutoencoder'
#         path = 'tb_logs_activation_sweep_no_prune/MNISTAutoencoder'
#         # path = 'tb_logs_activation_sweep_big_sweep_pruned/MNISTAutoencoder'
#         path = 'tb_logs_kl_sweep/MNISTAutoencoder'
#         folders_logs = glob.glob(path + '/*')
#         folders_logs
            
#         data = get_data(folders_logs)
#         data
#         data.dropna(inplace=True)
#         x_col = 'MODEL.LOSS.klDiv'
#         target_col = 'mse_us_tst'
#         data = data[[x_col, target_col]]
#         data = sort_by_mean(data, x_col, target_col)
#         # print(data)
#         # data, mapper = map_categoricals(data, 'ACTIVATION')
#         # fig, ax = plt.subplots(1,3, figsize=(15,5))

#         # hyper_param_chart(data, x_col,  target_col, normalize_y=True, ax=ax[0])

#         # hyperparam_error_bar_chart(data, ax[1], x_col, target_col)

#         # bin_plot(data, bins=6, ax=ax[2], x_col=x_col, target_col=target_col)
#         # plt.tight_layout()



#         data
#     x_col = 'MODEL.LOSS.klDiv'
#     target_col = 'mse_us_tst'
#     fig, ax = plt.subplots(1,1, figsize=(7,3))
#     ax2 = ax.twinx()
#     for i in range(len(data)):
#         x = data[x_col].iloc[i]
#         y = data[target_col].iloc[i]

#         # log scale x
#         x = np.log10(data[x_col]).values
#         x = (x - x.min()) / (x.max() - x.min())
#         x = x[i]

#         # normalize x
#         # x = (x - data[x_col].min()) / (data[x_col].max() - data[x_col].min())

#         # normalize y
#         y = (y - data[target_col].min()) / (data[target_col].max() - data[target_col].min())

#         ax.plot([0,1], [x, y], color=plt.cm.RdBu(y))
#         n_yticks = 3
#         ax.set_yticks(np.linspace(0, 1, n_yticks),
#                         np.round(np.linspace(
#                             np.log10(data[x_col].min()), 
#                             np.log10(data[x_col].max()),
#                                 n_yticks
#                                 ), 2))
        
        
#         ax2.set_yticks(np.linspace(0, 1, n_yticks), 
#                     np.round(np.linspace(data[target_col].min(), data[target_col].max(), n_yticks), 2))
        
#         ax.set_xticks([0,1], ['KL Divergence', 'MSE'])
#         ax.set_xlim(-.1, 1.1)
#         ax.set_ylabel('KL Divergence weight (log scale)')
#         ax2.set_ylabel('MSE')
#         fig.suptitle('KL Divergence Weight vs MSE')
#         ax.grid()

#     plt.savefig('assets/kl_weight_sweep.png', dpi=300)

#     """

# Determine joint connectivity
with tabs['Determine joint connectivity']:
    import numpy as np
    import streamlit as st
    import pandas as pd
    import plotly.graph_objects as go
    import time

    # set page layout to wide
    # st.set_page_config(layout="wide")

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



# Download data
with tabs['Download data']:
    from subpages.downloading_data import download_data_instruction
    download_data_instruction()

with tabs['links for later']:
    from subpages.links_for_later import links_for_later
    links_for_later()

