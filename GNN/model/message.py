from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import ipdb
import math
import time


class GRUMessagePasser(nn.Module):
    """relation-wise GRU for message passing"""
    def __init__(self, inp_dim, num_rels, dropout=0) -> None:
        super(GRUMessagePasser, self).__init__()
        self.edge_update = nn.Embedding(num_rels, inp_dim)
        self.edge_reset = nn.Embedding(num_rels, inp_dim)
        self.edge_candidate = nn.Embedding(num_rels, inp_dim)
        self.update_projection = nn.Linear(inp_dim, inp_dim, bias=True)
        self.reset_projection = nn.Linear(inp_dim, inp_dim, bias=True)
        self.candidate_projection = nn.Linear(inp_dim, inp_dim, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, edges):
        update_vector = self.dropout(self.sigmoid(self.edge_update(edges.data["type"]) * edges.data["edge_feat"] + self.update_projection(edges.src["node_feat"])))
        candidate_vector = self.sigmoid(self.edge_reset(edges.data["type"]) * edges.data["edge_feat"] + self.reset_projection(edges.src["node_feat"]))
        candidate_vector = self.tanh(self.edge_candidate(edges.data["type"]) * edges.data["edge_feat"] + self.candidate_projection(candidate_vector * edges.src["node_feat"]))
        message = update_vector * candidate_vector + (1 - update_vector) * edges.src["node_feat"]
        return {"h": message}


class AddMessagePasser(nn.Module):
    """Add Operator for message passing"""
    def __init__(self, inp_dim, num_rels, dropout=0) -> None:
        super(AddMessagePasser, self).__init__()
        self.edge_emb = nn.Embedding(num_rels, inp_dim)
        self.message_projection = nn.Linear(inp_dim, inp_dim, bias=True)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, edges):
        message = self.dropout(self.relu(self.message_projection(self.edge_emb(edges.data["type"]) + edges.src["node_feat"] + edges.data["edge_feat"])))
        return {"h": message}


class MulMessagePasser(nn.Module):
    """Mul Operator for message passing"""
    def __init__(self, inp_dim, num_rels, dropout=0) -> None:
        super(MulMessagePasser, self).__init__()
        self.edge_emb = nn.Embedding(num_rels, inp_dim)
        self.message_projection = nn.Linear(inp_dim, inp_dim, bias=True)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, edges):
        message = self.dropout(self.relu(self.message_projection(self.edge_emb(edges.data["type"]) * edges.src["node_feat"] * edges.data["edge_feat"])))
        return {"h": message}