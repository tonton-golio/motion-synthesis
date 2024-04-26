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