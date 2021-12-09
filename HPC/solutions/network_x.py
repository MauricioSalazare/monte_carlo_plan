import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

grid_info = pd.read_csv("../../data/processed_data/network_data/Lines_34.csv")

G = nx.Graph()
for _, (from_, to_, r) in grid_info[['FROM', 'TO', 'R']].iterrows():
    G.add_edge(int(from_), int(to_), length= r * 100)

voltages = np.random.randn((G.nodes.__len__()))
node_voltages = dict(zip(range(1, G.nodes.__len__() + 1), voltages.round(2)))
nx.set_node_attributes(G, node_voltages, "voltages")

#%%
fig, ax = plt.subplots(1,1,figsize=(10,10))
nx.draw_kamada_kawai(G, with_labels=True, font_size=7, node_size=150, node_color="skyblue", ax=ax)

#%%
fig, ax = plt.subplots(1,1,figsize=(10,10))
nx.draw_kamada_kawai(G, with_labels=True, font_size=7, node_size=150, node_color="skyblue", ax=ax)

#%%
labels = nx.get_node_attributes(G, 'voltages')

fig, ax = plt.subplots(1,1,figsize=(10,10))
nx.draw_kamada_kawai(G, with_labels=True, labels=labels, font_size=7, node_size=150, node_color="skyblue", ax=ax)
