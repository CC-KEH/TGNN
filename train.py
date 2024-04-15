import os
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import scipy.sparse as sp
import torch.nn.functional as F
import torch.optim as optim

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


def main():
    feat_d = 50
    dropout = 0.4
    hidden=64
    epochs = 10
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    
    
    # Data Loading
    # Assuming each row in the labels DataFrame now represents a sequence of graphs
    labels = pd.read_csv("dynamic_influence_train_set_sequence.csv")
    
    # dynamic_graph is the name of the sequence of graphs
    # dyanmic_graphs are the directories containing theie specific graph files
    
    labels.columns = ["dynamic_graph","node","infl"]
    gs = labels.graph_sequence.unique()

    sam = round(len(gs)/5)
    msk = [i for i in range(0,sam)]
    test_graphs = [gs[i] for i in msk]

    train_graphs = [gs[i] for i in range(len(gs)) if i not in msk ] 
    val_graphs = [train_graphs[i] for i in msk] 
    train_graphs = [train_graphs[i] for i in range(len(train_graphs)) if i not in msk] 

    traind = labels[labels.graph.isin(train_graphs)]
    testd = labels[labels.graph.isin(test_graphs)]
    vald = labels[labels.graph.isin(val_graphs)]
    
    # Read the graphs
    graph_dic = {}
    for g in gs:
        G = pd.read_csv("train/"+g+".txt",header=None,sep=" ")
        nodes = set(G[0].unique()).union(set(G[1].unique()))
        graph_dic[g] = sp.coo_matrix((G[2], (G[1], G[0])), shape=(len(nodes), len(nodes))).toarray()

    # Model Definition
    model = TemporalGNN(feat_d, hidden, int(hidden/2), int(hidden/4), dropout).to(device)

    # Training Loop
    for epoch in range(epochs):    
        model.train()
        train_loss = []

        for i in range(0,len(traind),batch_size):
            adj_sequence_batch = list()
            feature_sequence_batch = list()
            y_batch = list()
            idx_batch = list()

            for u in range(i,min(len(traind),i+batch_size) ):
                row  = traind.iloc[u]

                adj_sequence = [graph_dic[g] for g in row["graph_sequence"]]
                feature_sequence = [torch.zeros(adj.shape[0],feat_d) for adj in adj_sequence]

                for st in row["node"].split(","):
                    for features in feature_sequence:
                        features[int(st),:] = 1 

                adj_sequence_batch.append(adj_sequence)
                y_batch.append(row["infl"])
                idx_batch.extend([u-i]*adj_sequence[0].shape[0])
                feature_sequence_batch.append(feature_sequence)

            adj_sequence_batch = [sparse_mx_to_torch_sparse_tensor(sp.block_diag(adj_sequence)).to(device) for adj_sequence in adj_sequence_batch]
            feature_sequence_batch = [torch.cat(feature_sequence, dim=0).to(device) for feature_sequence in feature_sequence_batch]
            y_batch = torch.FloatTensor(y_batch).to(device)

            idx_batch_ = torch.LongTensor(idx_batch).to(device)

            optimizer.zero_grad()
            output = model(adj_sequence_batch, feature_sequence_batch, idx_batch_).squeeze()

            loss_train = F.mse_loss(output, y_batch)

            loss_train.backward(retain_graph=True)
            optimizer.step()

            train_loss.append(loss_train.data.item()/output.size()[0])

        # Validation
        model.eval()
        val_loss = []
        
        for i in range(0,len(vald),batch_size):
            adj_sequence_batch = list()
            feature_sequence_batch = list()
            y_batch = list()
            idx_batch = list()
        
            for u in range(i,min(len(vald),i+batch_size) ):
                row  = vald.iloc[u]
        
                adj_sequence = [graph_dic[g] for g in row["graph_sequence"]]
                feature_sequence = [torch.zeros(adj.shape[0],feat_d) for adj in adj_sequence]
        
                for st in row["node"].split(","):
                    for features in feature_sequence:
                        features[int(st),:] = 1 
        
                adj_sequence_batch.append(adj_sequence)
                y_batch.append(row["infl"])
                idx_batch.extend([u-i]*adj_sequence[0].shape[0])
                feature_sequence_batch.append(feature_sequence)
        
            adj_sequence_batch = [sparse_mx_to_torch_sparse_tensor(sp.block_diag(adj_sequence)).to(device) for adj_sequence in adj_sequence_batch]
            feature_sequence_batch = [torch.cat(feature_sequence, dim=0).to(device) for feature_sequence in feature_sequence_batch]
            y_batch = torch.FloatTensor(y_batch).to(device)
        
            idx_batch_ = torch.LongTensor(idx_batch).to(device)
        
            output = model(adj_sequence_batch, feature_sequence_batch, idx_batch_).squeeze()
        
            loss_val = F.mse_loss(output, y_batch)
        
            val_loss.append(loss_val.data.item()/output.size()[0])