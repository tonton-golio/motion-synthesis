import streamlit as st
import os
from glob import glob
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with st.sidebar:
    log_dirs = ['logs/mnistVAEoptuna/', 'MNIST_VAE', 'MNIST_Diffusion', 'CIFAR10_Diffusion']
    log_dir = st.selectbox('Select log directory', log_dirs)



def make_df(path_outer = 'logs/mnistVAEoptuna/'):

    log_dirs = glob('../mnist_latent_diffusion/'+path_outer + '*/')

    data = {}
    for dir in log_dirs:
        num = dir.split('/')[-2].split('_')[-1]
        print(dir)
        # check if hparams.yaml exists
        if not os.path.exists(dir + 'hparams.yaml'):
            print('hparams.yaml not found')
            continue

        with open(dir + 'hparams.yaml') as file:
            hparams = yaml.load(file, Loader=yaml.FullLoader)

        data[num] = hparams

    df = pd.DataFrame(data).T
    df.dropna(inplace=True)

    return df

df = make_df(log_dir)
df

# fig 1
plt.style.use('seaborn-v0_8-paper')
fig = plt.figure(figsize=(10, 6))
for ld in sorted(df['latent_dim'].unique()):
    df_ld = df[df['latent_dim'] == ld]
    kl = df_ld['klDivWeight']
    mse = df_ld['mse_us_tst']
    # print(f'Latent dim: {ld}')
    # print('KL Div Weight:', kl)
    # print('MSE:', mse)
    # print()
    plt.scatter(df_ld['klDivWeight'], df_ld['mse_us_tst'], label=str(ld))

plt.legend(title='Latent dim')
plt.xlabel('Kullback-Leibler divergence weight')
plt.ylabel('MSE (unscaled) on test set')
plt.xscale('log')
# plt.yscale('log')
# plt.grid()
# plt.xlim( 1e-6, 1e-4,)
# plt.savefig('assets/mnistVAEoptuna_klDivWeight_vs_mse_us_tst.png')
st.pyplot(fig)


# fig 2
fig = plt.figure(figsize=(10, 6))
sns.heatmap(df.corr())
st.pyplot(fig)



# fig 3: clustering
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



df.dropna(inplace=True)
X = df.values
y = df['latent_dim'].values
mse = df['mse_us_tst'].values


reducer = umap.UMAP()
embedding = reducer.fit_transform(X)

# cluster
kmeans = KMeans(n_clusters=3, random_state=0).fit(embedding)
fig = plt.figure(figsize=(10, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], c=mse, cmap='viridis', s=10)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar()


plt.legend()
st.pyplot(fig)



"""
import os, sys, glob



import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np

def get_data(folders):
    data = {}
    for folder in folders:
        folder_num = int(folder.split('/')[-1].split('_')[-1])
        with open(folder + '/hparams.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            data[folder_num] = params

    df = pd.DataFrame(data).T
    df.sort_index(inplace=True)
    return df

def map_categoricals(df, col_name):
    mapper = {k: i for i, k in enumerate(df[col_name].unique())}
    df[col_name+'_mapped'] = df[col_name].map(mapper)
    return df, mapper

def sort_by_mean(df, x_col, target_col):
    mean = df.groupby(x_col).mean()

    # sort by mean
    category_names = mean[target_col].sort_values().index
    # print(category_names)
    #sort df by category_names
    # print(df)
    df = df.set_index(x_col).loc[category_names].reset_index()
    # print(df)
    mapper = {k: i for i, k in enumerate(df[x_col].unique())}
    df[x_col+'_mapped'] = df[x_col].map(mapper)
    return df

def hyper_param_chart(df, x_col, target_col, ax, normalize_y=True):
    # we normalize all columns, and plot lines for each row
    # scale to between 0 and 1
    

    y = df[target_col].values.flatten()
    x = df[x_col+'_mapped'].values.flatten()
    try:
        x = (x - x.min()) / (x.max() - x.min())
    except:
        print('x scaled failed')
        x = x
    
    try:
        y_scaled = (y - y.min()) / (y.max() - y.min())
    except:
        print('y scaled failed')
        y_scaled = y
    if normalize_y:
        y = y_scaled
    # print('x', x)
    # print('y', y)
    
    ax2 = ax.twinx()
    ax.set_ylim(-.1, 1.1)
    ax2.set_ylim(-.1, 1.1)
    # print(mapper)
    # ax.set_yticks(np.linspace(0, 1, len(mapper)), mapper.keys())

    # color by y value
    for i in range(len(x)):
        ax.plot([0, 1], [x[i], y[i]], color=plt.cm.RdBu(y_scaled[i]))
        # ax.plot(range(2), [x[i], y[i]], label=i)

    ax.grid()

def hyperparam_error_bar_chart(data, ax,  x_col='ACTIVATION',  target_col='mse_unscaled_test'):
    # print(data)
    # data = sort_by_mean(data, x_col, target_col)
    mean = data.groupby(x_col).mean()
    std = data.groupby(x_col).std()
    try:
        mean_scaled = ((mean - mean.min()) / (mean.max() - mean.min()))[target_col]
    except:
        print('mean scaled failed')
        mean_scaled = mean[target_col]

    # sort by mean
    category_names = mean[target_col].sort_values().index
    mean = mean.loc[category_names]
    std = std.loc[category_names]

    # print(mean)
    # print(std)
    for i, (m, s) in enumerate(zip(mean[target_col], std[target_col])):
        ax.errorbar(m, i, xerr=s, fmt='o', color=plt.cm.viridis(mean_scaled[i]))

    ax.set_yticks(range(len(mean.index)), mean.index)
    # return mean, std

# count how many times each activation function yields a binned mse
def bin_plot(data, bins, ax,  x_col='ACTIVATION', target_col='mse_unscaled_test'):
    # data = sort_by_mean(data, x_col, target_col)

    bins = np.linspace(data[target_col].min(), data[target_col].max(), bins)
    bin_labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
    counts = {}
    for i in range(len(data)):
        
        act = data[x_col].iloc[i]
        val = data[target_col].iloc[i]
        # print(act, val)
        if act not in counts:
            counts[act] = np.zeros(len(bins)-1)
        for j in range(len(bins)-1):
            if val >= bins[j] and val < bins[j+1]:
                counts[act][j] += 1
                break

    
    
    # print(counts)
    # Ensure every activation has a count for each bin, even if it's 0
    all_bins = np.zeros(len(bins)-1)
    for act in counts:
        counts[act] = counts[act] + all_bins
    colors = plt.cm.RdBu(np.linspace(0,1, len(bins)))
    # Prepare data for stacked bar chart
    
    for i, (act, vals) in enumerate(counts.items()):
        # print(act)
        bottom = 0
        
        for c, v in zip(colors, vals):
            ax.barh(i+bottom-.25, v, height=.1, color=c)
            bottom += .1

        

    # show the bins in a legend
    labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
    handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(bins)-1)]
    ax.legend(handles, labels)
    ax.set_xlim(-1, 12)

    ax.set_yticks(range(len(counts)), counts.keys())


if __name__ == '__main__':
    path = 'tb_logs_activation_sweep_no_prune_100_epochs/MNISTAutoencoder'
    path = 'tb_logs_activation_sweep_no_prune/MNISTAutoencoder'
    # path = 'tb_logs_activation_sweep_big_sweep_pruned/MNISTAutoencoder'
    path = 'tb_logs_kl_sweep/MNISTAutoencoder'
    folders_logs = glob.glob(path + '/*')
    folders_logs
        
    data = get_data(folders_logs)
    data
    data.dropna(inplace=True)
    x_col = 'MODEL.LOSS.klDiv'
    target_col = 'mse_us_tst'
    data = data[[x_col, target_col]]
    data = sort_by_mean(data, x_col, target_col)
    # print(data)
    # data, mapper = map_categoricals(data, 'ACTIVATION')
    # fig, ax = plt.subplots(1,3, figsize=(15,5))

    # hyper_param_chart(data, x_col,  target_col, normalize_y=True, ax=ax[0])

    # hyperparam_error_bar_chart(data, ax[1], x_col, target_col)

    # bin_plot(data, bins=6, ax=ax[2], x_col=x_col, target_col=target_col)
    # plt.tight_layout()



    data
x_col = 'MODEL.LOSS.klDiv'
target_col = 'mse_us_tst'
fig, ax = plt.subplots(1,1, figsize=(7,3))
ax2 = ax.twinx()
for i in range(len(data)):
    x = data[x_col].iloc[i]
    y = data[target_col].iloc[i]

    # log scale x
    x = np.log10(data[x_col]).values
    x = (x - x.min()) / (x.max() - x.min())
    x = x[i]

    # normalize x
    # x = (x - data[x_col].min()) / (data[x_col].max() - data[x_col].min())

    # normalize y
    y = (y - data[target_col].min()) / (data[target_col].max() - data[target_col].min())

    ax.plot([0,1], [x, y], color=plt.cm.RdBu(y))
    n_yticks = 3
    ax.set_yticks(np.linspace(0, 1, n_yticks),
                      np.round(np.linspace(
                          np.log10(data[x_col].min()), 
                          np.log10(data[x_col].max()),
                            n_yticks
                            ), 2))
    
    
    ax2.set_yticks(np.linspace(0, 1, n_yticks), 
                   np.round(np.linspace(data[target_col].min(), data[target_col].max(), n_yticks), 2))
    
    ax.set_xticks([0,1], ['KL Divergence', 'MSE'])
    ax.set_xlim(-.1, 1.1)
    ax.set_ylabel('KL Divergence weight (log scale)')
    ax2.set_ylabel('MSE')
    fig.suptitle('KL Divergence Weight vs MSE')
    ax.grid()

plt.savefig('assets/kl_weight_sweep.png', dpi=300)

"""