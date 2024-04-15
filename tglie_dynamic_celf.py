import numpy as np
import scipy.sparse as sp
import torch
from queue import PriorityQueue
import torch.nn as nn


class TemporalGNN(nn.Module):
    def __init__(self, n_feat, n_hidden, dropout):
        super(TemporalGNN, self).__init__()
        self.gnn = GNN_skip_small(n_feat, n_hidden, int(n_hidden/2), int(n_hidden/4), dropout)
        self.lstm = nn.LSTM(n_hidden, n_hidden, batch_first=True)
        self.fc = nn.Linear(n_hidden, 1)
        
    def forward(self, adj, x_in, idx):
        gnn_outputs = []
        for adj, x in zip(adj, x_in):
            gnn_output = self.gnn(adj, x, idx)
            gnn_outputs.append(gnn_output.unsqueeze(0))
        gnn_outputs = torch.cat(gnn_outputs, dim=0)
        
        lstm_output, _ = self.lstm(gnn_outputs)
        lstm_output = lstm_output[-1]
        
        output = self.fc(lstm_output)
        return output

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def dynamic_celf_TGNN(model, K, graph_sequence, feature_sequence, device):
    """
    Dynamic CELF function using TemporalGNN for influence prediction.
    """
    # Initialize the set of selected nodes and the priority queue
    selected_nodes = set()
    queue = PriorityQueue()

    # For each node in the graph
    for node in range(feature_sequence[0].shape[0]):
        # Prepare the input data for the TemporalGNN model
        adj_sequence = [graph.to(device) for graph in graph_sequence]
        feature_sequence_node = [features.clone().to(device) for features in feature_sequence]
        for features in feature_sequence_node:
            features[node, :] = 1

        # Compute the predicted influence of the node using the TemporalGNN model
        predicted_influence = model(adj_sequence, feature_sequence_node, torch.tensor([node], device=device)).item()

        # Add the node to the priority queue with priority equal to the negative predicted influence
        queue.put((-predicted_influence, node))

    # While the set of selected nodes is smaller than the K
    while len(selected_nodes) < K:
        # Pop the node with the highest priority from the queue
        _, node = queue.get()

        # If the node is not in the set of selected nodes
        if node not in selected_nodes:
            # Add the node to the set of selected nodes
            selected_nodes.add(node)

            # For each node in the graph
            for other_node in range(feature_sequence[0].shape[0]):
                # If the other node is not the same as the node
                if other_node != node:
                    # Prepare the input data for the TemporalGNN model
                    adj_sequence = [graph.to(device) for graph in graph_sequence]
                    feature_sequence_node = [features.clone().to(device) for features in feature_sequence]
                    for features in feature_sequence_node:
                        features[other_node, :] = 1

                    # Compute the predicted influence of the other node given the selected nodes
                    predicted_influence = model(adj_sequence, feature_sequence_node, torch.tensor([other_node], device=device)).item()

                    # Update the priority of the other node in the queue
                    queue.put((-predicted_influence, other_node))

    # Return the set of selected nodes
    return selected_nodes

def main():
    # Testing Function
    # Load the data and apply the CELF algorithm
    pass


