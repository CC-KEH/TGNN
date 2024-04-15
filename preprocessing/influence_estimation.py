import os
from tqdm import tqdm
import time
import glob
import pandas as pd
import networkx as nx
import random
import numpy as np
from multiprocessing import Pool, Lock
from improved_greedy import improved_greedy_algorithm_dynamic

random.seed(1)
lock = Lock()

# Check if the file exists, and if not, create it
if not os.path.exists("../influence_labels.csv"):
    open("../influence_labels.csv", 'w').close()


def process_graph(dg):
    seed_size = random.randint(1, 5)  # generate a random seed size between 1 and 5
    S, S_influence_spread = improved_greedy_algorithm_dynamic(dg, seed_size)
    result = dg+','+str(S)+','+str(S_influence_spread)+"\n"
    
    with lock:
        with open("../influence_labels.csv", "a") as labels:
            labels.write(result)

if __name__ == "__main__":
    t = time.time() 

    os.chdir("../data/sim_graphs/train")
    with Pool() as p:
        results = list(tqdm(p.imap(process_graph, glob.glob("*")), total=len(glob.glob("*"))))

    print("Time taken:", time.time() - t)


# labels = open("../influence_labels.csv","a")

# def process_graph(dg):
#     seed_size = random.randint(1, 5)  # generate a random seed size between 1 and 5
#     S, S_influence_spread = improved_greedy_algorithm_dynamic(dg, seed_size)
#     result = dg+','+str(S)+','+str(S_influence_spread)+"\n"
    
#     # Write the result to the file
#     with open("../influence_labels.csv", "a") as labels:
#         labels.write(result)
        
# if __name__ == "__main__":
#     t = time.time() 

#     os.chdir("../data/sim_graphs/train")
#     with Pool() as p:
#         results = list(tqdm(p.imap(process_graph, glob.glob("*")), total=len(glob.glob("*"))))

#     with open("../influence_labels.csv", 'w') as labels:
#         labels.writelines(results)


# os.chdir("../data/sim_graphs/train")
# for dg in tqdm(glob.glob("*")):
#     seed_size = random.randint(1, 5)  # generate a random seed size between 1 and 5
#     S, S_influence_counts = improved_greedy_algorithm_dynamic(dg, seed_size)
#     labels.write(dg+','+str(S)+','+str(S_influence_counts)+"\n")
    
# labels.close()