import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import ipdb


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class DirectedLayer(nn.Module):
    def __init__(self, inp_dim, out_dim, activation=None, self_included=False):
        super(DirectedLayer, self).__init__()
        self.self_included = self_included
        self.activation = activation if activation else Identity()

        self.linear_head = nn.Linear(inp_dim * (2 if self_included else 1), out_dim)
        self.linear_tail = nn.Linear(inp_dim * (2 if self_included else 1), out_dim)

    def forward(self, g:dgl.DGLGraph, efeat:torch.tensor):
        g = g.local_var()

        g.edata['head'], g.edata['tail'] = torch.chunk(efeat, 2, -1)

        rg = dgl.reverse(g, copy_edata=True)

        g.update_all(fn.copy_e('head', 'm'), fn.mean('m', 'efeat_g'))
        rg.update_all(fn.copy_e('tail', 'm'), fn.mean('m', 'efeat_g'))

        g.ndata['nfeat'] = (g.ndata.pop('efeat_g') + rg.ndata.pop('efeat_g')) / 2
        virtual_head = g.edata['head'].mean(dim=0)
        virtual_tail = g.edata['tail'].mean(dim=0)

        g.apply_edges(
            lambda edges: {
                # 'head': torch.cat([edges.src['nfeat'], (edges.data['head'] + virtual_head) / 2], -1) if self.self_included else edges.src['nfeat'],
                # 'tail': torch.cat([edges.dst['nfeat'], (edges.data['tail'] + virtual_tail) / 2], -1) if self.self_included else edges.dst['nfeat'],
                'head': torch.cat([edges.src['nfeat'], edges.data['head']], -1) if self.self_included else edges.src['nfeat'],
                'tail': torch.cat([edges.dst['nfeat'], edges.data['tail']], -1) if self.self_included else edges.dst['nfeat'],
            })

        g.edata['head'] = self.linear_head(g.edata['head'])
        g.edata['tail'] = self.linear_tail(g.edata['tail'])
        efeat = torch.cat([g.edata['head'], g.edata['tail']], -1)
        
        return self.activation(efeat)
