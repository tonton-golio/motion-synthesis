import streamlit as st
import json
import os
import numpy as np

# Load the JSON data from file
def load_json(file_path='todo.json'):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, 'r') as f:
        return json.load(f)
    
def sort_json(data):
    # takes the completed tasks and moves them to the bottom
    data.sort(key=lambda x: x['completed'])
    return data

# Save the JSON data to file
def save_json(data, file_path='todo.json'):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

# Function to add a new entry
def make_new_entry(data):
    new_task = st.text_input("New task")

    # check if exists
    for item in data:
        if item['task'] == new_task:
            st.error("Task already exists.")
            
            return

    if new_task:
        subtasks = st.text_input("Subtasks (comma separated)").split(',')
        data.append({
            'task': new_task,
            'completed': False,
            'old': False,  # this is for the 'old' tasks that are not relevant anymore, but we don't want to delete them
            'subtasks': [{'task': subtask.strip(), 'completed': False} for subtask in subtasks if subtask]
        })
        save_json(data)
        

# Function to update an existing entry
def update_entry(data, old=False):
    for idx, item in enumerate(data):
        # st.write(f"{idx+1}. {item}")
        if item['old'] != old:
            continue
        cols = st.columns([2, 3, 1])

        with cols[0]:
            prev_status = item['completed']
            if st.checkbox(item['task'], item['completed']) != prev_status:
                item['completed'] = not item['completed']
        
        with cols[1]:
            if item['subtasks']:
                for sub_idx, sub_item in enumerate(item['subtasks']):
                    prev_status = sub_item['completed']
                    if st.checkbox(sub_item['task'], sub_item['completed'], key=f"{idx}-{sub_idx}"+ ('-old' if old else '')) != prev_status:
                        sub_item['completed'] = not sub_item['completed']
        
        with cols[2]:
            prev_old = item['old']
            if st.checkbox("Old", item['old'], key=f"{idx}-old" + ('-old' if old else '')) != prev_old:
                item['old'] = not item['old']


        if not old:
            save_json(data)
        st.divider()
        # break

# Main interface for the TODO list
def todo_interface():
    
    data = load_json()
    data = sort_json(data)
    make_new_entry(data)


    st.divider()
    if not data:
        st.write("No tasks found.")
    else:
        # st.write("### TODO List")
        update_entry(data)

        with st.expander("Old tasks"):
            update_entry(data, old=True)

    
def todo_interface2():
    st.write("""
    We have a couple different categories of tasks:
             
    ### App
        - [] Make comprehensive TODO list
        - [x] Implement motion synthesis inference mode in the app
        - [] Clean up diffusion theory page
        - [] add math to KL divergence page
        - [] fix MNIST diffusion page
        - [] Build out of MNIST VAE page
        - [] fix folder numbers in assets and assets_produced
             

    ### Manuscript
        - [] improved data-flow chart
        - [] combine figure 3 + 4, and 5 + 6
        - [] Rewrite section on **Parameterizing the noise schedule**, it should be clear that we first tried shannon, but went for KL instead.
        - [] remove mnist images in theory section
        - [] add subsection CLIP to theory section
        - [] remove subsubsections in Metrics

    ### MNIST Tuning
        - [x] MNIST diffusion model
        - [] MNIST VAE -> 3 latent spaces with different KL weights --> just 1
        - [] MNIST LD 

    ### Pose encoding
        - [] cluster poses

    ### Motion Tuning
        - [] Tune VAE6
        - [] tune LD for both VAE6 and VAE1
     
    ### Model Functionality
        - [] implement loss scaled percentage plot
        - [] Enable motion LD continue from checkpoint
    
    """)
    

if __name__ == "__main__":
    st.title("Thesis TODO List")
    todo_interface()