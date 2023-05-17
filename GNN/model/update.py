import torch
import torch.nn as nn


class LSTMEdgeUpdator(nn.Module):
    """LSTM Operator for edge updating"""
    def __init__(self, out_dim, dropout=0) -> None:
        super(LSTMEdgeUpdator, self).__init__()
        self.lstm = nn.LSTM(input_size=out_dim, hidden_size=out_dim, num_layers=1, dropout=dropout)
    
    def forward(self, edges):
        x_, (_, c_) = self.lstm(edges.data["edge_feat"].unsqueeze(0), (edges.src["node_feat"].unsqueeze(0), edges.data["edge_query"].unsqueeze(0)))
        torch.cuda.empty_cache()
        return {"edge_feat": x_.squeeze(0), "edge_query": c_.squeeze(0)}


class MLPEdgeUpdator(nn.Module):
    """MLP Operator for edge updating"""
    def __init__(self, out_dim, dropout=0) -> None:
        super(MLPEdgeUpdator, self).__init__()
        self.feat_proj = nn.Linear(out_dim * 3, out_dim)
        self.query_proj = nn.Linear(out_dim * 3, out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, edges):
        edge_feat = self.dropout(self.feat_proj(torch.cat([edges.data["edge_feat"], edges.src["node_feat"], edges.data["edge_query"]], dim=-1)))
        edge_query = self.dropout(self.query_proj(torch.cat([edges.data["edge_feat"], edges.src["node_feat"], edges.data["edge_query"]], dim=-1)))
        return {"edge_feat": edge_feat, "edge_query": edge_query}