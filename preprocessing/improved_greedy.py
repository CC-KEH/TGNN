import networkx as nx
import numpy as np
import glob

def construct_graph(G, A):
    # Start with a copy of the original graph
    Gd = G.copy()

    # For each vertex in the graph
    for v in Gd.nodes:
        # If the vertex is active
        if v in A:
            # For each neighbor of the vertex
            neighbors = list(Gd.neighbors(v))  # Create a copy of the neighbors
            for u in neighbors:
                # The edge from v to u will be in the graph with probability p_v_u
                p_v_u = Gd[v][u]['weight']  # assuming the edge weight represents the probability
                if np.random.rand() > p_v_u:
                    Gd.remove_edge(v, u)

    # For each vertex, add a self-edge with probability 1
    for v in Gd.nodes:
        Gd.add_edge(v, v)

    return Gd

def compute_influence(Gd, A, v):
    # Start with the active set
    C = A.copy()

    # Add the vertex v to the active set
    C.add(v)

    # C = A U {v}
    
    # Initialize the influence of v
    phi_v = len(C)

    # Repeat until no more vertices can be added to C
    while True:
        # Find all vertices that are connected to C and not already in C
        H = set()
        for u in C:
            H.update(set(Gd.neighbors(u)) - C)

        # If no more vertices can be added, break the loop
        if not H:
            break

        # Add the vertices in H to C and update the influence of v
        C.update(H)
        phi_v += len(H)

    return phi_v


def improved_greedy_algorithm(G, k, R = 10000):
    A = set()
    influences = {}

    for i in range(k):
        for j in range(R):
            Gd = construct_graph(G, A)

            for v in set(Gd.nodes) - A:
                if v not in influences:
                    influences[v] = 0

                influences[v] += compute_influence(Gd, A, v)

        # Normalize the influences and find the vertex with the maximum influence
        for v in influences:
            influences[v] /= R

        max_v = max(influences, key=influences.get)
        A.add(max_v)

    # Calculate the total influence spread of the seed set
    total_influence_spread = sum(influences[v] for v in A)

    # Create a dictionary mapping each node in A to its influence spread
    A_influence_spread = {v: influences[v] for v in A}

    return A, total_influence_spread, A_influence_spread


def improved_greedy_algorithm_dynamic(dg_dir, k):
    total_influence_spread = 0
    A_influence_spread = {}

    # Read each static graph in the dynamic graph
    for g_file in glob.glob(f"{dg_dir}/*"):
        print("Reading:",g_file)
        
        G = nx.read_edgelist(g_file, data=(('weight', float),))  # specify that edge data includes a weight
        A_for_G, influence_spread_for_G, A_influence_spread_for_G = improved_greedy_algorithm(G, k, R=1000)  # run the existing algorithm on the static graph

        # Add the influence spread of the seed set to the total influence spread
        total_influence_spread += influence_spread_for_G

        # Update the influence spread of each node in the seed set
        for v in A_for_G:
            if v not in A_influence_spread:
                A_influence_spread[v] = 0
            A_influence_spread[v] += A_influence_spread_for_G[v]

        print(f"Result on {g_file}:", A_for_G, total_influence_spread)
        
    # Select the k nodes with the highest influence spread
    A = set(sorted(A_influence_spread, key=A_influence_spread.get, reverse=True)[:k])

    return A, total_influence_spread