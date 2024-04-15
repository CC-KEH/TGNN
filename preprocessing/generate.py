import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import networkx as nx
from dynamic_graph_generator import dynamic_graph_generator


train_directory = '/Users/arbas/Documents/Projects/Major/TGLIE/data/sim_graphs/train'
val_directory = '/Users/arbas/Documents/Projects/Major/TGLIE/data/sim_graphs/val'


# 100 Powerlaw graphs with 100-200 nodes.
# 50 Powerlaw graphs with 300-500 nodes.
# 100 Small-world graphs with 100-200 nodes.
# 50 Small-world graphs with 300-500 nodes.
# 100 Erdos-Renyi graphs with 100-200 nodes.
# 50 Erdos-Renyi graphs with 300-500 nodes.

# 60 - 20 - 20 split for training, validation and test sets.


if not os.path.exists(train_directory):
    os.makedirs(train_directory)

if not os.path.exists(val_directory):
    os.makedirs(val_directory)

generator = dynamic_graph_generator()

max_time_stamps = 10
min_time_stamps = 5

# Training

# Powerlaw
# generator.gen_new_graphs(100,200,5,max_time_stamps,train=True,g_type='powerlaw')
# generator.save_graphs(train_directory)

# generator.gen_new_graphs(300,500,5,max_time_stamps,train=True,g_type='powerlaw')
# generator.save_graphs(train_directory)

# generator.gen_new_graphs(100,200,5,min_time_stamps,train=True,g_type='powerlaw')
# generator.save_graphs(train_directory)

# generator.gen_new_graphs(300,500,5,min_time_stamps,train=True,g_type='powerlaw')
# generator.save_graphs(train_directory)


# # Small-world
# generator.gen_new_graphs(100,200,5,max_time_stamps,train=True,g_type='small-world')
# generator.save_graphs(train_directory)

# generator.gen_new_graphs(300,500,5,max_time_stamps,train=True,g_type='small-world')
# generator.save_graphs(train_directory)

# generator.gen_new_graphs(100,200,5,min_time_stamps,train=True,g_type='small-world')
# generator.save_graphs(train_directory)

# generator.gen_new_graphs(300,500,5,min_time_stamps,train=True,g_type='small-world')
# generator.save_graphs(train_directory)



# # Erdos-Renyi
# generator.gen_new_graphs(100,200,5,max_time_stamps,train=True,g_type='erdos_renyi')
# generator.save_graphs(train_directory)

# generator.gen_new_graphs(300,500,5,max_time_stamps,train=True,g_type='erdos_renyi')
# generator.save_graphs(train_directory)

# generator.gen_new_graphs(100,200,5,min_time_stamps,train=True,g_type='erdos_renyi')
# generator.save_graphs(train_directory)

# generator.gen_new_graphs(300,500,5,min_time_stamps,train=True,g_type='erdos_renyi')
# generator.save_graphs(train_directory)



# # Validation

# # Powerlaw
# generator.gen_new_graphs(100,200,15,max_time_stamps,train=False,g_type='powerlaw')
# generator.save_graphs(val_directory)

# generator.gen_new_graphs(300,500,15,max_time_stamps,train=False,g_type='powerlaw')
# generator.save_graphs(val_directory)

# generator.gen_new_graphs(100,200,15,min_time_stamps,train=False,g_type='powerlaw')
# generator.save_graphs(val_directory)

# generator.gen_new_graphs(300,500,15,min_time_stamps,train=False,g_type='powerlaw')
# generator.save_graphs(val_directory)


# # Small-world
# generator.gen_new_graphs(100,200,15,max_time_stamps,train=False,g_type='small-world')
# generator.save_graphs(val_directory)

# generator.gen_new_graphs(300,500,15,max_time_stamps,train=False,g_type='small-world')
# generator.save_graphs(val_directory)

# generator.gen_new_graphs(100,200,15,min_time_stamps,train=False,g_type='small-world')
# generator.save_graphs(val_directory)

# generator.gen_new_graphs(300,500,15,min_time_stamps,train=False,g_type='small-world')
# generator.save_graphs(val_directory)


# # Erdos-Renyi
# generator.gen_new_graphs(100,200,15,max_time_stamps,train=False,g_type='erdos_renyi')
# generator.save_graphs(val_directory)

# generator.gen_new_graphs(300,500,15,max_time_stamps,train=False,g_type='erdos_renyi')
# generator.save_graphs(val_directory)

# generator.gen_new_graphs(100,200,15,min_time_stamps,train=False,g_type='erdos_renyi')
# generator.save_graphs(val_directory)

# generator.gen_new_graphs(300,500,15,min_time_stamps,train=False,g_type='erdos_renyi')
# generator.save_graphs(val_directory)



os.chdir(train_directory)


# make them directed
for g in tqdm(glob.glob("*/*")):
    G = pd.read_csv(g, header=None, sep=" ")
    G.columns = ["node1", "node2", "w"]
    del G["w"]
    # make undirected directed
    tmp = G.copy()
    G = pd.DataFrame(np.concatenate([G.values, tmp[["node2", "node1"]].values]), columns=G.columns)
    G.columns = ["source", "target"]
    outdegree = G.groupby("target").agg('count').reset_index()
    outdegree.columns = ["target", "weight"]
    outdegree["weight"] = 1 / outdegree["weight"]
    outdegree["weight"] = outdegree["weight"].apply(lambda x: float('%s' % float('%.6f' % x)))
    G = G.merge(outdegree, on="target")
    G.to_csv(g, sep=" ", header=None, index=False)