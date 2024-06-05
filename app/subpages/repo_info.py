import os
import pandas as pd
import os, sys, glob
import streamlit as st

# counts the number of lines, words, characters, and functions in the repo
def count_in_file(file_path):
    # this could be done with wc -l, wc -w, wc -c, and grep -c 'def ' file_path
    # or with regex
    with open(file_path, 'r') as file:
        f = file.read()

    return dict(
        lines=f.count('\n'),
        words=len(f.split()),
        chars=len(f),
        functions=f.count('def '),
    )

def count_lines_in_folder(folder, exclude):
    
    total = {'lines': 0, 'words': 0, 'chars': 0, 'functions': 0}
    for root, _, files in os.walk(folder):
        
        # check each of the exclude strings in the root
        if any([e in root for e in exclude]):
            continue
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                counts = count_in_file(file_path)
                for key in counts:
                    total[key] += counts[key]

    return total

def count_info(folders = ["../app",  "../mnist_latent_diffusion",  "../motion_latent_diffusion"], 
         exclude = ['logs']):

    # List of folders to search
    data = {}
    for folder in folders:
        if os.path.exists(folder):
            count = count_lines_in_folder(folder, exclude)
            data[folder.split('../')[1]] = count
        else:
            print(f"Folder not found: {folder}")
    df = pd.DataFrame(data)
    df['Total'] = df.sum(axis=1)
    
    return df


# storage usage
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

import plotly.express as px

def groupby_and_pie(df, column, n=8):
    grouped = df.groupby(column)['size_gb'].sum().sort_values(ascending=False).astype(int)[:n]
    
    # Create pie chart
    fig = px.pie(grouped, values='size_gb', names=grouped.index, 
                title=column.split('_')[0].capitalize(),
                labels={'size_gb': 'Size (GB)', 'index': column},
                color=grouped.index)

    # Update legend title
    fig.update_layout(legend_title_text=column.split('_')[0].capitalize())
    
    # Adjust legend position and formatting
    fig.update_layout(
        legend=dict(
            title=dict(font=dict(size=14)),
            font=dict(size=12),
            orientation="h",  # horizontal orientation
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    return fig, grouped

