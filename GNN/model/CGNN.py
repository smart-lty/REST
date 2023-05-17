import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
import ipdb
import math
import time
from model.pna_utils import Identity
from model.message import GRUMessagePasser, AddMessagePasser, MulMessagePasser
from model.update import LSTMEdgeUpdator, MLPEdgeUpdator
from model.aggregate import PNAAggregator, MinAggregator, MaxAggregator, MeanAggregator


class CycleGNNLayer(nn.Module):
    def __init__(self, inp_dim, out_dim, num_rels, message="gru", aggregate="pna", update="lstm", activation=None, dropout=0) -> None:
        super(CycleGNNLayer, self).__init__()

        self.activation = activation or Identity()
        message_passer =  {   
            "gru": GRUMessagePasser(inp_dim=inp_dim, num_rels=num_rels, dropout=dropout), 
            "add": AddMessagePasser(inp_dim=inp_dim, num_rels=num_rels, dropout=dropout),
            "mul": MulMessagePasser(inp_dim=inp_dim, num_rels=num_rels, dropout=dropout)
                                }
        aggregater = {   
            "pna": PNAAggregator(inp_dim=inp_dim, out_dim=out_dim),
            "max": MaxAggregator(inp_dim=inp_dim, out_dim=out_dim),
            "min": MinAggregator(inp_dim=inp_dim, out_dim=out_dim),
            "mean": MeanAggregator(inp_dim=inp_dim, out_dim=out_dim)
                                }
        edge_updater = {   
            "lstm": LSTMEdgeUpdator(out_dim=out_dim, dropout=dropout), 
            "mlp": MLPEdgeUpdator(out_dim=out_dim, dropout=dropout)
                                }
        
        self.message = message_passer[message]
        self.aggregate = aggregater[aggregate]
        self.edge_update = edge_updater[update]
        
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, g:dgl.DGLGraph, efeat:torch.tensor, nfeat:torch.tensor, equery:torch.tensor):
        """cycle message passing for inductive subgraph reasoning"""
        g = g.local_var()

        g.edata["edge_feat"] = efeat
        g.ndata["node_feat"] = nfeat
        g.edata["edge_query"] = equery

        self.aggregate.avg_d = torch.mean(torch.log(g.out_degrees() + 1))

        g.update_all(self.message, self.aggregate)
        g.apply_edges(self.edge_update)

        return self.layer_norm(g.edata.pop("edge_feat")), self.layer_norm(g.ndata.pop("node_feat")), self.layer_norm(g.edata.pop("edge_query"))


class CycleGNN(nn.Module):
    def __init__(self, params, relation2id) -> None:
        super(CycleGNN, self).__init__()
        self.params = params
        self.relation2id = relation2id

        self.inp_dim = params.inp_dim
        self.hid_dim = params.emb_dim
        self.num_rels = params.num_rels

        self.dropout = nn.Dropout(params.dropout)

        self.query_embedding = nn.Embedding(self.num_rels, self.inp_dim)

        self.edge_query_projection = nn.Linear(self.inp_dim * 2, self.inp_dim)

        self.input_layer = CycleGNNLayer(
            inp_dim=self.inp_dim,
            out_dim=self.hid_dim,
            num_rels=self.num_rels,
            message=self.params.message,
            aggregate=self.params.aggregate,
            update=self.params.update,
            activation=F.relu,
            dropout=self.params.dropout
        )

        self.hidden_layer = nn.ModuleList()
        for _ in range(params.num_gcn_layers - 1):
            self.hidden_layer.append(
                CycleGNNLayer(
                    inp_dim=self.hid_dim,
                    out_dim=self.hid_dim,
                    num_rels=self.num_rels,
                    message=self.params.message,
                    aggregate=self.params.aggregate,
                    update=self.params.update,
                    activation=F.relu,
                    dropout=self.params.dropout
                )
            )
        
        self.edge_jk_connection = nn.Linear(params.num_gcn_layers * self.hid_dim, self.hid_dim)
        self.node_jk_connection = nn.Linear(params.num_gcn_layers * self.hid_dim, self.hid_dim)
        self.query_jk_connection = nn.Linear(params.num_gcn_layers * self.hid_dim, self.hid_dim)
        nn.init.xavier_uniform_(self.edge_jk_connection.weight)
        nn.init.xavier_uniform_(self.node_jk_connection.weight)
        nn.init.xavier_uniform_(self.query_jk_connection.weight)

        self.fc_layer = nn.Linear(self.hid_dim * 4, 1)
        nn.init.xavier_uniform_(self.fc_layer.weight)
    

    def forward(self, g: dgl.DGLGraph):
        # single-source initialization of all edge features
        efeat = torch.zeros((g.number_of_edges(), self.inp_dim)).to(self.params.device)
        queries = self.query_embedding(g.edata["type"][g.edata["target_edge"]])
        efeat[g.edata["target_edge"]] = queries
        
        ## ablation study: single-source initialization against full initialization
        # efeat = self.query_embedding(g.edata["type"])

        # initialization of all node features (hidden features)
        nfeat = torch.zeros((g.number_of_nodes(), self.inp_dim)).to(self.params.device)

        # initialization of all edge queries
        equery = queries.reshape(-1, self.inp_dim * 2)
        equery = equery[torch.cat([torch.ones(g.batch_num_edges()[i]) * i for i in range(g.batch_num_edges().shape[0])]).long().to(self.params.device)]
        equery = self.edge_query_projection(equery)

        # input layer
        efeat_o, nfeat_o, equery_o = self.input_layer(g, efeat, nfeat, equery)
        if self.params.residual:
            efeat = efeat + efeat_o
            nfeat = nfeat + nfeat_o
            equery = equery + equery_o
        else:
            efeat = efeat_o
            nfeat = nfeat_o
            equery = equery_o
        
        efeats = [efeat]
        nfeats = [nfeat]
        equeries = [equery]

        # hidden layer
        for layer in self.hidden_layer:
            efeat_o, nfeat_o, equery_o = layer(g, efeat, nfeat, equery)
            
            if self.params.residual:
                efeat = efeat + efeat_o
                nfeat = nfeat + nfeat_o
                equery = equery + equery_o
            else:
                efeat = efeat_o
                nfeat = nfeat_o
                equery = equery_o
                
            efeats.append(efeat)
            nfeats.append(nfeat)
            equeries.append(equery)
        
        efeats = self.edge_jk_connection(torch.cat(efeats, dim=-1))
        nfeats = self.node_jk_connection(torch.cat(nfeats, dim=-1))
        equeries = self.query_jk_connection(torch.cat(equeries, dim=-1))

        target_edge_feats = efeats[g.edata["target_edge"]].reshape(g.batch_size, 2, -1)
        target_query_feats = equeries[g.edata["target_edge"]].reshape(g.batch_size, 2, -1)
        
        target_nodes = g.edges()[0][g.edata["target_edge"]].reshape(g.batch_size, 2)

        target_head_feats = nfeats[target_nodes[:, 0]]
        target_tail_feats = nfeats[target_nodes[:, 1]]

        right_scores = self.fc_layer(torch.cat([target_edge_feats[:, 0, :], target_query_feats[:, 0, :], target_head_feats, target_tail_feats], dim=-1))
        left_scores = self.fc_layer(torch.cat([target_edge_feats[:, 1, :], target_query_feats[:, 1, :], target_tail_feats, target_head_feats], dim=-1))
        
        return torch.max(right_scores, left_scores)