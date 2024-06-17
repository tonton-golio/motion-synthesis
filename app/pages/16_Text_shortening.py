import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

st.set_page_config(layout="centered")
df = pd.read_csv("../motion_latent_diffusion/text_backup/texts.csv")
df.rename(columns={"Unnamed: 0": "fname"}, inplace=True)

path_animations= '../stranger_repos/HumanML3D/HumanML3D/animations/'

"# Text shortening"

cols = st.columns(2)
with cols[1]:
    idx = st.slider("Select idx", 0, 10, 7)
with cols[0]:
    
    st.write("""
            Our text strings vary in length and quality. They are written in natural language, by human annotators, so they are all unique.
        We have strings which look like this (for the idx selected):
    """)
    st.write(df.iloc[idx].text.split("\n"))

    action = df.iloc[idx].action
    st.write(f"LLM generated action: `{action}`")

with cols[1]:
    fname = df.iloc[idx].fname
    fname  # 010727.txt
    st.video(path_animations + fname.split('.')[0] + '.mp4')


st.divider()
"### System prompt"
'''
To shorten the text descriptions to a single action-describing word, we could extract the verbs from the text and set up some simple heuristics. But instead we opt for prompting an LLM with the full text and asking it to generate a single word.

It is crucial that we engineer an appropriate system prompt for the LLM, so that it understands the task at hand. We will use the following prompt:
'''

sys_promt1 = "`You will be given 2-4 short texts describing the same motion sequence. Your task is to paraphrase the text as 1 verb describing the motion.`"

sys_prompt2="`You will be given 2-4 short texts describing the same motion sequence. Your task is to identify the common action and paraphrase it as one verb that best describes the overall motion. reply with a verb in its present tense dictionary form, i.e. to 'verb'. You are to reply with onely 1 word. Use common/simple verbs.`"

sys_prompt3 = "`You will be given 2-4 short texts describing the same motion sequence. Identify the common action and paraphrase it as a single verb that best describes the overall motion. Reply with only one simple verb, such as 'walk', 'jump', 'lift', etc.`"

cols = st.columns(3)
with cols[0]:
    st.write(sys_promt1)

with cols[1]:
    st.write(sys_prompt2)

with cols[2]:
    st.write(sys_prompt3)

text_ex = ["the person bend over and pick some thing up.",
"a peraon bends over, using the right leg to bear weight while kicking back his left leg, and picks something up with his right hand.",
"a person picks up something with their right hand."]
"""
given the texts:
"""
text_ex

"GPT-3-turbo generated actions:"
cols = st.columns(3)
with cols[0]:
    st.code("Bending")

with cols[1]:
    st.code("to bend")

with cols[2]:
    st.code("pick")

"GPT-4 generated actions:"
cols = st.columns(3)
with cols[0]:
    st.code("stoop")

with cols[1]:
    st.code('To pick')

with cols[2]:
    st.code("lift")

"GPT-4o generated actions:"
cols = st.columns(3)
with cols[0]:
    st.code("Retrieve")

with cols[1]:
    st.code('To “pick”')

with cols[2]:
    st.code("pick")


st.divider()

"## Results"
vc = df.action.value_counts()



# fig, ax = plt.subplots(figsize=(8, 4))

# # Group value counts into bins and count the number of occurrences in each bin
# bins= [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 207]
# vc_binned = pd.cut(vc, bins=bins)
# # vc_binned
# vc_binned_counts = vc_binned.value_counts().sort_index()

# # Plot the binned counts
# vc_binned_counts.plot(kind='bar', ax=ax)

# # Set plot title and labels
# plt.title("Number of actions with a given frequency")
# plt.ylabel("Number of actions")
# plt.xlabel("Frequency")
# plt.yscale('log')
# plt.grid(True)
# # bins
# bin_labels = [f"{b.left}-{b.right}" for b in vc_binned_counts.index]# if b < 10 else f"{b.left}+"]
# # bin_labels
# plt.xticks(np.arange(len(bins)-1), bin_labels, rotation=0)
# plt.tight_layout()

# # Display the plot
# st.pyplot(fig)

st.write("The most common actions are:")
vc_copy = vc.copy()
vc_copy.sort_values(ascending=False, inplace=True)

n_other = 16
tmp = vc_copy[n_other:].sum()

vc_copy = vc_copy[:n_other]
vc_copy['Other'] = tmp

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
vc_copy.plot(kind='bar', ax=ax)
plt.xticks(rotation=45)
plt.yticks([1000, 3000, 15000])
plt.title("Most common actions")
plt.ylabel("Frequency")
plt.xlabel("Action")
plt.grid()
plt.tight_layout()
st.pyplot(fig)


# st.write("The least common actions are:")
# vc[-10:]

# st.divider()

n_api_requests = 30_623
n_tokens = 3_564_773
cost = 1.82 # USD
f"""
We have decrease our number of identifiers from `{len(df)}` to `{len(vc)}`. But we still want to reduce the number of unique `actions` further.

This is great value for money, as we have made `{n_api_requests}` API requests and processed `{n_tokens}` tokens for a total cost of `{cost}` USD.
"""

st.divider()
"""
I noticed that the action `walks` and `walk` are both in the list of actions. So check if `s` at the end of any actions
"""
with st.expander("Click to see the actions"):
    list_of_actions = vc.index
    action_mapper = {}
    # list_of_actions
    cols = st.columns(3)
    col_idx = 0

    list_of_actions_end_s = [a for a in list_of_actions if a[-1] == 's']
    for a in list_of_actions_end_s:
        if a[:-1] in list_of_actions:  # if we have the singular form
            # a, "------>" ,a[:-1]
            cols[col_idx%len(cols)].write(f"`{a}`" + r'$\rightarrow$' + f"`{a[:-1]}`")
            action_mapper[a] = a[:-1]
            col_idx += 1

        else:
            cols[col_idx%len(cols)].write(a)
            col_idx += 1

st.divider()
"""
I also notice that we have two word actions, lets look at these. We see preposition `to`,
"""

with st.expander("Click to see the actions"):
    list_of_actions_2words = [a for a in list_of_actions if len(a.split()) > 1]
    cols = st.columns(3)
    col_idx = 0

    to_remove = ['to ', ' to']
    # len(list_of_actions_2words), len(to_remove)
    for a in list_of_actions_2words:
        col_idx_prev = col_idx
        for r in to_remove:
            if r in a:
                if a.replace(r, '') in list_of_actions:
                    if col_idx_prev == col_idx:
                        col_idx += 1
                        cols[col_idx%len(cols)].write(f"`{a}`" + r'$\rightarrow$' + f"`{a.replace(r, '')}`")
                        action_mapper[a] = a.replace(r, '')
                    
                else:
                    if col_idx_prev == col_idx:             
                        cols[col_idx%len(cols)].write(f"**{a}**")
                        col_idx += 1
        else:
            if col_idx_prev == col_idx:
                cols[col_idx%len(cols)].write(f"{a}")
                col_idx += 1

    
st.divider()
"""
Look for strange characters in the actions, I saw `verb: dance` above, so lets check for any non-alphabetic characters. (hmm including spaces).

Check if the action with special characters removed is in the list of actions.
"""

with st.expander("Click to see the actions"):
    from string import ascii_letters
    permissible_chars = set(ascii_letters + ' ')

    list_of_actions_special_chars = [a for a in list_of_actions if not all(c in permissible_chars for c in a)]
    cols = st.columns(3)
    col_idx = 0
    for a in list_of_actions_special_chars:
        if ''.join([c for c in a if c in permissible_chars]) in list_of_actions:
            cols[col_idx%len(cols)].write(f"`{a}`" + r'$\rightarrow$' + f"`{''.join([c for c in a if c in permissible_chars])}`")
            action_mapper[a] = ''.join([c for c in a if c in permissible_chars])
            col_idx += 1
        else:
            cols[col_idx%len(cols)].write(a)
            col_idx += 1

df['action_mapped'] = df['action'].apply(lambda x: action_mapper.get(x, x))
# df

st.divider()

vc_smaller = df['action_mapped'].value_counts()
vc_smaller.sort_values(ascending=False, inplace=True)

f"""
After removing the actions, we now have `{len(vc_smaller)}` unique actions.

Next, we should consider that we have all actions and their mirror, we can look if the action and mirrorer action have the same label, if not choose the most common one?
"""
df['mirrored'] = df['fname'].apply(lambda x: x[0] == 'M')
df['file_number'] = df['fname'].apply(lambda x: x.replace('M', '').split('.')[0])
df[df['file_number'] == '011439']

"""
How often do they share the same label?
"""
action_mapper2 = {}
count_same_label = 0
count_diff_label = 0
for fn in df['file_number'].unique():
    tmp = df[df['file_number'] == fn]
    if tmp['action_mapped'].nunique() == 1:
        count_same_label += 1
        action_mapper2[fn] = tmp['action_mapped'].iloc[0]
    else:
        act1 = tmp['action_mapped'].value_counts().index[0]
        act2 = tmp['action_mapped'].value_counts().index[1]
        
        # which one is more common
        if vc_smaller[act1] > vc_smaller[act2]:
            action_mapper2[fn] = act1
        else:
            action_mapper2[fn] = act2


        count_diff_label += 1

f"""
In `{count_same_label}` cases the action and its mirror have the same label. In `{count_diff_label}` cases they have different labels.
"""
# action_mapper2

df['action_mapped_2'] = df['file_number'].apply(lambda x: action_mapper2[x])

vc_smaller_2 = df['action_mapped_2'].value_counts()
vc_smaller_2.sort_values(ascending=False, inplace=True)

vc_smaller_2_len = len(vc_smaller_2)
f"""
After removing the actions, we now have `{vc_smaller_2_len}` unique actions.
"""
vc_smaller_2

list_of_actions_4print = ', '.join(vc_smaller_2.index)
tmp = vc_smaller_2[:10].sum()
vc_smaller_2 = vc_smaller_2[:10]
vc_smaller_2['Other'] = tmp

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
vc_smaller_2.plot(kind='bar', ax=ax)
plt.xticks(rotation=45)
# plt.yticks([1000, 3000, 15000])

plt.title("Most common actions")
plt.ylabel("Frequency")
plt.xlabel("Action")
plt.grid()
plt.tight_layout()

st.pyplot(fig)

n_clusters = '?'
f"""
We could also try to cluster the actions, to do so, we need to ask an LLM to cluster the `{vc_smaller_2_len}`.
"""

"""
A good prompt for the LLM clustering model lets us group actions that are semantically similar. We could use the following prompt:

```
I will provide a list of actions about 300. The list contains actions, some of which are semantically similar. Your task is to cluster the list of actions sematically. I want about 30 clusters.

please provide your answer as:
clusters = {
    0 : {
        'description': 'description of the cluster',
        'actions': ['action1', 'action2', ...]
    },
    1 : { ... },
}
"""+f"""
list of actions:

{list_of_actions_4print}
```
Since its just a single forward pass, we can afford to use a big model.
"""

clusters = {
    0: {
        'description': 'Basic locomotion',
        'actions': ['walk', 'jog', 'run', 'sprint', 'stroll', 'stride', 'strut', 'saunter', 'meander', 'amble', 'circumambulate', 'wander', 'roam', 'pace', 'march', 'step', 'advance']
    },
    1: {
        'description': 'Rapid movements',
        'actions': ['dash', 'sprint', 'hurry', 'rush', 'charge', 'accelerate', 'speedwalk'] + ['sidestep']
    },
    2: {
        'description': 'Jumping and leaping',
        'actions': ['jump', 'leap', 'hop', 'bounce', 'vault', 'skip']
    },
    3: {
        'description': 'Changing direction and orientation',
        'actions': ['turn', 'rotate', 'pivot', 'twirl', 'spin', 'swivel']
    },
    4: {
        'description': 'Lifting and carrying',
        'actions': ['lift', 'carry', 'hoist', 'tote', 'transfer']
    },
    5: {
        'description': 'Throwing and launching',
        'actions': ['throw', 'toss', 'hurl', 'pitch', 'cast', 'fling', 'launch']
    },
    6: {
        'description': 'Hand movements',
        'actions': ['wave', 'clap', 'gesture', 'point', 'clutch', 'grip', 'grasp', 'hold', 'shake', 'pat', 'touch', 'tap']
    },
    7: {
        'description': 'Lower body movements',
        'actions': ['kick', 'stomp', 'step', 'tap', 'squatting', 'kneel', 'crouch', 'bend', 'squat', 'stoop', 'duck']
    },
    8: {
        'description': 'Body balance and control',
        'actions': ['balance', 'tiptoe', 'crawl', 'creep', 'sidle', 'slither']
    },
    9: {
        'description': 'Swimming and water movements',
        'actions': ['swim', 'paddle', 'splash', 'dive', 'float']
    },
    10: {
        'description': 'Dancing and rhythmic movements',
        'actions': ['dance', 'sway', 'waltz', 'twirl', 'spin', 'shimmy', 'sashay'] + ['moonwalk', 'swagger']
    },
    11: {
        'description': 'Climbing and descending',
        'actions': ['climb', 'ascend', 'descend', 'crawl', 'shimmy', 'scale']
    },
    12: {
        'description': 'Falling and recovering',
        'actions': ['fall', 'tumble', 'stumble', 'trip', 'collapse', 'crash', 'slip', 'slide']
    },
    13: {
        'description': 'Fighting and defensive actions',
        'actions': ['punch', 'kick', 'strike', 'block', 'defend', 'fight', 'hit', 'slap']
    },
    14: {
        'description': 'Picking and placing',
        'actions': ['pick', 'place', 'put', 'position', 'arrange', 'rearrange']
    },
    15: {
        'description': 'Catching and holding',
        'actions': ['catch', 'grab', 'snatch', 'hold', 'clutch', 'seize']
    },
    16: {
        'description': 'Pulling and pushing',
        'actions': ['pull', 'push', 'tug', 'drag', 'shove']
    },
    17: {
        'description': 'Stretching and bending',
        'actions': ['stretch', 'bend', 'flex', 'extend', 'reach']
    },
    18: {
        'description': 'Gesturing and signaling',
        'actions': ['wave', 'gesture', 'signal', 'salute', 'nod', 'wink']
    },
    19: {
        'description': 'Writing and drawing',
        'actions': ['write', 'draw', 'sketch', 'scribble', 'doodle']
    },
    20: {
        'description': 'Using tools and instruments',
        'actions': ['cut', 'chop', 'saw', 'hammer', 'drill', 'screw', 'manipulate', 'use']
    },
    21: {
        'description': 'Communicating and expressing',
        'actions': ['talk', 'speak', 'shout', 'whisper', 'sing', 'laugh', 'cry']
    },
    22: {
        'description': 'Eating and drinking',
        'actions': ['eat', 'drink', 'sip', 'gulp', 'chew', 'swallow']
    },
    23: {
        'description': 'Cleaning and grooming',
        'actions': ['wash', 'clean', 'scrub', 'wipe', 'brush', 'groom', 'bathe']
    },
    24: {
        'description': 'Observing and inspecting',
        'actions': ['look', 'glance', 'peek', 'scan', 'observe', 'inspect']
    },
    25: {
        'description': 'Handling and manipulating objects',
        'actions': ['move', 'shift', 'adjust', 'arrange', 'rearrange']
    },
    26: {
        'description': 'Standing and sitting',
        'actions': ['stand', 'sit', 'rise', 'get up', 'lower', 'descend']
    },
    27: {
        'description': 'Resting and relaxing',
        'actions': ['sit', 'lie', 'rest', 'pause', 'relax']
    },
    28: {
        'description': 'Performing physical exercises',
        'actions': ['exercise', 'workout', 'train', 'lift', 'stretch', 'run', 'jog', 'walk']
    },
    29: {
        'description': 'Miscellaneous actions',
        'actions': ['think', 'contemplate', 'meditate', 'ponder', 'consider']
    }
}


f"""
GPT-4o has clustered the actions into `{len(clusters)}` clusters. Here are the clusters:

```python
{clusters}
```
"""

group_mapper = {}
for c in clusters:
    for a in clusters[c]['actions']:
        group_mapper[a] = clusters[c]['description']



df['action_group'] = df['action_mapped_2'].apply(lambda x: group_mapper.get(x, 'Other'))

fig, ax = plt.subplots(1, 1, figsize=(35, 5))
df['action_group'].value_counts().plot(kind='bar', ax=ax)
plt.xticks(rotation=45)
plt.title("Action groups")
plt.ylabel("Frequency")
plt.xlabel("Action group")
plt.grid()
plt.tight_layout()
st.pyplot(fig)

"""
Nice, then we will just label encode both the actions and the action groups and we are good to go!


"""
action_group_mapper = dict(zip(range(len(df.action_group.unique())), df.action_group.unique()))
action_group_mapper_inv = {v: k for k, v in action_group_mapper.items()}

# do the same for action_mapped_2

action_mapped_2_mapper = dict(zip(range(len(df.action_mapped_2.unique())), df.action_mapped_2.unique()))
action_mapped_2_mapper_inv = {v: k for k, v in action_mapped_2_mapper.items()}
# action_mapped_2_mapper_inv

df['action_group_num'] = df['action_group'].apply(lambda x: action_group_mapper_inv[x])
df['action_mapped_2_num'] = df['action_mapped_2'].apply(lambda x: action_mapped_2_mapper_inv[x])

df

if st.radio("Save the data?", ['No', 'Yes']) == 'Yes':
    # save df
    df.to_csv("../motion_latent_diffusion/text_backup/texts_grouped.csv", index=False)

    # save the mappers
    np.save("../motion_latent_diffusion/text_backup/action_group_mapper.npy", action_group_mapper)
    np.save("../motion_latent_diffusion/text_backup/action_mapped_2_mapper.npy", action_mapped_2_mapper)