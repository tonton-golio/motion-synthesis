import streamlit as st
import os
import glob
import subprocess

# Title and introduction
"""
# Make GIF

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