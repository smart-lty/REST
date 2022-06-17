import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import ipdb
from .layers import DirectedLayer

class DirectedEdgeConv(nn.Module):
    def __init__(self, params, relation2id):
        super(DirectedEdgeConv, self).__init__()
        self.params = params
        self.relation2id = relation2id
        
        self.inp_dim = params.inp_dim
        self.hid_dim = params.emb_dim
        self.num_rels = params.num_rels
        self.dropout = params.dropout

        self.device = params.device

        self.rel_features = nn.Embedding(self.num_rels, self.inp_dim * 2)
        nn.init.xavier_uniform_(self.rel_features.weight, gain=nn.init.calculate_gain('relu'))

        self.input_layer = DirectedLayer(self.inp_dim,
                         self.hid_dim,
                         activation=F.relu,
                         self_included=True
                         )
        
        self.hidden_layer = nn.ModuleList()
        for _ in range(params.num_gcn_layers - 1):
            self.hidden_layer.append(
                                DirectedLayer(self.hid_dim,
                                self.hid_dim,
                                activation=F.relu,
                                self_included=True
                                )
                            )
        self.jk_connection = nn.Linear(params.num_gcn_layers * self.hid_dim * 2, self.hid_dim * 2)
        self.linear = nn.Linear(self.hid_dim * 2, 1)

    def forward(self, g:dgl.DGLGraph):
        repr = self.get_representation(g)
        scores = self.linear(repr)
        return scores
    
    def get_representation(self, g:dgl.DGLGraph):
        efeat = self.propogate(g)
        repr = efeat[g.edata["target_edge"]]
        return repr

    def propogate(self, g:dgl.DGLGraph):
        efeat = self.rel_features(g.edata['type'])

        features = []
        
        # input layer
        efeat = self.input_layer(g, efeat)
        features.append(efeat)

        # hidden layer
        for layer in self.hidden_layer:
            efeat = layer(g, efeat)
            features.append(efeat)

        # jumping knowledge connection
        features = torch.cat(features, dim=-1)

        if self.params.using_jk:
            return self.jk_connection(features)
        return efeat
