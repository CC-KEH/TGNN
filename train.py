import os
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import scipy.sparse as sp
import torch.nn.functional as F
import torch.optim as optim
import random
import time

class TemporalGNN(nn.Module):
    def __init__(self, n_feat, n_hidden, dropout):
        super(TemporalGNN, self).__init__()
        self.gnn = GNN_skip_small(n_feat, n_hidden, int(n_hidden/2), int(n_hidden/4), dropout)
        self.lstm = nn.LSTM(n_hidden, n_hidden, batch_first=True)
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, adj_sequence_batch, feature_sequence_batch, idx, lengths):
        gnn_outputs = []
        for adj_sequence, feature_sequence in zip(adj_sequence_batch, feature_sequence_batch):
          sequence_outputs = []
          for adj, features in zip(adj_sequence, feature_sequence):
              gnn_output = self.gnn(adj, features, idx)
              sequence_outputs.append(gnn_output)
          sequence_outputs = torch.stack(sequence_outputs)
          gnn_outputs.append(sequence_outputs.unsqueeze(0))
        gnn_outputs = torch.cat(gnn_outputs, dim=0)

        # Pack the sequence
        gnn_outputs = nn.utils.rnn.pack_padded_sequence(gnn_outputs, lengths, batch_first=True, enforce_sorted=False)

        lstm_output, _ = self.lstm(gnn_outputs)

        # Unpack the sequence
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)

        # Use the last valid output for each sequence
        lstm_output = lstm_output[torch.arange(lstm_output.size(0)), lengths - 1]

        output = self.fc(lstm_output)
        return output

def forward(self, adj_sequence_batch, feature_sequence_batch, idx, lengths):
    gnn_outputs = []
    for adj_sequence, feature_sequence in zip(adj_sequence_batch, feature_sequence_batch):
        sequence_outputs = []
        for adj, features in zip(adj_sequence, feature_sequence):
            gnn_output = self.gnn(adj, features, idx)
            sequence_outputs.append(gnn_output)
        sequence_outputs = torch.stack(sequence_outputs)
        gnn_outputs.append(sequence_outputs.unsqueeze(0))
    gnn_outputs = torch.cat(gnn_outputs, dim=0)

    # Pack the sequence
    gnn_outputs = nn.utils.rnn.pack_padded_sequence(gnn_outputs, lengths, batch_first=True, enforce_sorted=False)

    lstm_output, _ = self.lstm(gnn_outputs)

    # Unpack the sequence
    lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)

    # Use the last valid output for each sequence
    lstm_output = lstm_output[torch.arange(lstm_output.size(0)), lengths - 1]

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




def main():
    v="1"
    feat_d = 50
    dropout = 0.4
    hidden=64
    epochs = 10
    batch_size = 16
    early_stop = 20
    learn = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Loading
    traind = pd.read_csv("train_labels.csv")
    traind.columns = ["dynamic_graph","node","infl"]
    # Read the graphs
    graph_dic = {}
    train_graphs_dir = "train/train"
    # No of nodes inside dynamic graph may change over time. 
    max_nodes_dic = {}
    for dynamic_graph in os.listdir(train_graphs_dir):
        max_nodes = 0
        dynamic_graph_dir = os.path.join(train_graphs_dir, dynamic_graph)
        for static_graph_file in os.listdir(dynamic_graph_dir):
            G = pd.read_csv(os.path.join(dynamic_graph_dir, static_graph_file), header=None, sep=" ")
            nodes = set(G[0].unique()).union(set(G[1].unique()))
            if len(nodes) > max_nodes:
                max_nodes = len(nodes)
        max_nodes_dic[dynamic_graph] = max_nodes

    for dynamic_graph in os.listdir(train_graphs_dir):
        graph_dic[dynamic_graph] = []
        dynamic_graph_dir = os.path.join(train_graphs_dir, dynamic_graph)
        max_nodes = max_nodes_dic[dynamic_graph]
        for static_graph_file in os.listdir(dynamic_graph_dir):
            G = pd.read_csv(os.path.join(dynamic_graph_dir, static_graph_file), header=None, sep=" ")
            adj = sp.coo_matrix((G[2], (G[1], G[0])), shape=(max_nodes, max_nodes)).toarray()
            graph_dic[dynamic_graph].append(adj)

    # Model Definition
    model = TemporalGNN(feat_d, hidden, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learn)
    error_log = open("errors/train_error"+str(v)+".txt","w")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    bigstart = time.time()
    print("Training....")
    # Training Loop
    for epoch in range(epochs):    
        model.train()
        train_loss = []
        max_node = max(int(node) for nodes in traind['node'] for node in nodes.split(','))
        # Check the number of nodes in your graph
        num_nodes = max(max(adj.shape) for adj_sequence in graph_dic.values() for adj in adj_sequence)
        # If max_node is greater than or equal to num_nodes, you need to increase num_nodes
        if max_node >= num_nodes:
            num_nodes = max_node + 1  # Increase the number of nodes to be larger than the maximum node index
        #------------- train for one epoch
        for i in range(0,len(traind),batch_size):
            adj_sequence_batch = list()
            feature_sequence_batch = list()
            y_batch = list()
            idx_batch = list()
            #------------ create batch
            for u in range(i,min(len(traind),i+batch_size) ):
                row  = traind.iloc[u]
                adj_sequence = graph_dic[row["dynamic_graph"]]
                feature_sequence = [torch.zeros(adj_mat.shape[0],feat_d) for adj_mat in adj_sequence]
                
                for st in row["node"].split(","):
                  for features in feature_sequence:
                    features[int(st),:] = torch.ones(feat_d)
                adj_sequence_batch.append(adj_sequence)
                y_batch.append(row["infl"])
                idx_batch.extend([u-i]*adj_sequence[0].shape[0])
                feature_sequence_batch.append(feature_sequence)

            lengths = torch.LongTensor([len(adj_sequence) for adj_sequence in adj_sequence_batch])
            _, sorted_indices = lengths.sort(descending=True)
            lengths = lengths[sorted_indices]
            
            adj_sequence_batch = [adj_sequence_batch[i] for i in sorted_indices]

            # Flatten feature_sequence_batch into a list of 2D tensors
            feature_sequence_batch = [feature_sequence_batch[i] for i in sorted_indices]


            # Convert y_batch and idx_batch into tensors
            y_batch = torch.tensor(y_batch).to(device)
            idx_batch_ = torch.LongTensor(idx_batch).to(device)

            output = model(adj_sequence_batch, feature_sequence_batch, idx_batch_, lengths).squeeze()

            # loss_train = F.mse_loss(output, y_batch)

            # loss_train.backward(retain_graph=True)
            # optimizer.step()

            # train_loss.append(loss_train.data.item()/output.size()[0])

if __name__=="__main__":
  main()