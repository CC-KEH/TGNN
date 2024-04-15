import os
import networkx as nx
import numpy as np
import sys
import random
from tqdm import tqdm
import glob

class dynamic_graph_generator():
    
    def __init__(self):
        """
        Code adjusted from https://github.com/FFrankyy/FINDER
        """
        self.TestSet = []
        self.TrainSet = []
        
    def gen_new_graphs(self, min_nodes, max_nodes, graphs_count, sequence_length, train = True, g_type =  'random', w_type = 'degree'):
        if train:
            print(f'\ngenerating new {g_type} training graphs...')
        else:
            print(f'\ngenerating new {g_type} validation graphs...')
            
        sys.stdout.flush()
        self.g_type = g_type
        self.w_type = w_type
        self.graphs_count = graphs_count
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.sequence_length = sequence_length
        # sequence_length is total timestamps or the number of graphs in dynamic graph.
        
        for i in tqdm(range(graphs_count)):
            g_sequence = [self.gen_graph() for _ in range(sequence_length)]
            if train:
                self.TrainSet.append(g_sequence)
            else:
                self.TestSet.append(g_sequence)

    def gen_graph(self):
        cur_n = np.random.randint(self.max_nodes - self.min_nodes + 1) + self.min_nodes
        if self.g_type == 'erdos_renyi':
            g = nx.erdos_renyi_graph(n=cur_n, p=0.15)
        
        elif self.g_type == 'small-world':
            g = nx.connected_watts_strogatz_graph(n=cur_n, k=8, p=0.1)
            
        elif self.g_type == 'powerlaw':
            x = random.randint(0,1)
            if x<1:
                g = nx.powerlaw_cluster_graph(n=cur_n, m=5,p=0.05,seed=1)
            else:
                g = nx.barabasi_albert_graph(n=cur_n, m=5,seed=1)
                
        ### random weight
        if self.w_type == 'random':
            weights = {}
            for node in g.nodes():
                weights[node] = random.uniform(0,1)
                
        # ### degree weight
        elif self.w_type  == 'degree':
            degree = nx.degree(g)
            maxDegree = max(dict(degree).values())
            weights = {}
            for node in g.nodes():
                weights[node] = degree[node]/maxDegree
                
        elif self.w_type == 'degree_noise':
            degree = nx.degree(g)
            mu = np.mean(list(dict(degree).values()))
            std = np.std(list(dict(degree).values()))
            weights = {}
            for node in g.nodes():
                episilon = np.random.normal(mu, std, 1)[0]
                weights[node] = 0.5*degree[node] + episilon
                if weights[node] < 0.0:
                    weights[node] = -weights[node]
            maxDegree = max(weights.values())
            for node in g.nodes():
                weights[node] = weights[node] / maxDegree
    
        return g

    def save_graphs(self, path):
        for i, g_sequence in enumerate(self.TrainSet):
            os.makedirs(f"{path}/{self.g_type[:2]}_{self.sequence_length}_DG_{i}", exist_ok=True)
            for j, g in enumerate(g_sequence):
                nx.write_edgelist(g, f"{path}/{self.g_type[:2]}_{self.sequence_length}_DG_{i}/graph_{j}.txt")       
        
    def load_graphs(self, path):
        for l in glob.glob(f"{path}/*"):
            self.TrainSet.append(nx.read_edgelist(l))