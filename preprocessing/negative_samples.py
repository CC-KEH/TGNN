import pandas as pd
import networkx as nx
import random
import glob
from improved_greedy import compute_influence,construct_graph

seed_size = 5
neg_samples = 20
directory = "../data/sim_graphs/train"  # Update this to your actual directory

# Read influence_labels.csv
x = pd.read_csv("influence_labels.csv",header=None)
x.columns = ["graph","node","infl"]
x["len"] = x.node.apply(lambda x: len(x.split(",")))

labels = open('../dynamic_influence_train_set.csv','a')
# Get unique graphs



for dg_dir in glob.glob(f"{directory}/*"):  # Loop over directories in the specified directory
    print(dg_dir)
    Gs = [nx.read_edgelist(g_file, data=(('weight', float),)) for g_file in glob.glob(f"{dg_dir}/*")]
    tmp = x[x.graph==dg_dir.split('/')[-1]]  # Get rows for the current graph

    # Loop over each row in tmp
    for i, row in tmp.iterrows():
        seeds = row["node"].split(",")
        # Draw randomly other seed sets of the same length
        for k in range(neg_samples):
            neg_seeds = set(random.sample(Gs.nodes(), row["len"]))
            counter = 0
            while neg_seeds == seeds:
                counter += 1
                neg_seeds = set(random.sample(Gs.nodes(), row["len"]))
                if counter > 10:
                    print("stuck")

            # Calculate influence spread for the negative seeds
            Gd = construct_graph(Gs, neg_seeds)
            neg_spread = sum(compute_influence(Gd, neg_seeds, v) for v in neg_seeds)
            labels.write(dg_dir + ',"' + '","'.join([str(ng) for ng in neg_seeds]) + '",' + str(neg_spread) + '\n')
        labels.write(row["graph"] + ',"' + row["node"] + '",' + str(row['infl']) + '\n')
