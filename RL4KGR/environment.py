import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import time
import ipdb
sys.path.append("/home/tyliu/code/inductive-master/")
sys.path.append("/home/tyliu/code/inductive-master/GNN/")
main_dir = os.path.relpath(os.path.dirname(os.path.abspath(__file__)))
from GNN.utils import *

class KnowledgeGraph(object):
    """
    Reinforcement Learning for Inductive Knowledge Graph Link Prediction.
    
    Steps:
    =================================================================================================
    1. Initialize subgraphs with given triples.
    =================================================================================================
    while not stop_flag:
    =================================================================================================
        2. Encode subgraphs with Pre-training / New DirectedEdgeConv GNNs into tensors.
    =================================================================================================
        3. Observe the environment. Return the action space of next step. (include stop action.)
    =================================================================================================
        4. Reasoning with the policy network. Return the action probablities of each possible action.
    =================================================================================================
        5. Update the subgraphs with the reasoning results.
    =================================================================================================

    TODO: 
    1. Reward  2. Action stop: detailed analysis
    """

    def __init__(self, raw_data_paths, model=None, add_transpose_rels=False, included_relations=None, readout="single", load_model=None):
        """Initial Parameters. Process the raw data into DGLGraph Object."""
        self.file_path = raw_data_paths
        self.add_transpose_rels = add_transpose_rels
        self.included_relations = included_relations
        self.readout = readout
        
        adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(raw_data_paths, included_relations)
        
        self.num_rels = len(adj_list)

        if add_transpose_rels:
            adj_list_t = [adj.T for adj in adj_list]
            adj_list += adj_list_t
        
        self.graph = ssp_multigraph_to_dgl(adj_list)
        self.triplets = triplets
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.id2entity = id2entity
        self.id2relation = id2relation

        self.reward_func = model
    
    def initialze_subgraphs(self, triples):
        """initialize subgraphs with target triples. """
        subgraphs = []
        labels = []
        for triple in triples:
            subgraph = self.graph.subgraph((triple[0], triple[1]))
            heads, tails, eids = subgraph.edges('all')
            # indicator for edges between the head and tail
            indicator1 = torch.logical_and(heads==0, tails==1)
            # indicator for edges to be predicted
            indicator2 = torch.logical_and(subgraph.edata['type'] == triple[2], indicator1)
            
            if self.readout == "single":
                if indicator2.sum() == 0:
                    subgraph.add_edge(0, 1)
                    subgraph.edata['type'][-1] = torch.tensor(self.num_rels).type(torch.LongTensor)
                    subgraph.edata["_ID"][-1] = torch.tensor(-1).type(torch.LongTensor)
                else:
                    subgraph.edata['type'][eids[indicator2]] = torch.tensor(self.num_rels).type(torch.LongTensor)
            
            elif self.readout == "agg":
                if indicator2.sum() == 0:
                    subgraph.add_edge(0, 1)
                    subgraph.edata['type'][-1] = torch.tensor(triple[2]).type(torch.LongTensor)
                    subgraph.edata["_ID"][-1] = torch.tensor(-1).type(torch.LongTensor)
            else:
                raise NotImplementedError
            subgraphs.append(subgraph)
            labels.append(triple[2])

        self.subgraphs = subgraphs
        self.labels = labels
        
    def state_encoding(self, steps):
        """encode current subgraphs into Pytorch Tensors."""
        batch_subgraphs = dgl.batch(self.subgraphs)
        self.subgraph_embeddings = self.reward_func.get_representation(batch_subgraphs, steps)
        
    def observe(self):
        """Observe the environment. Return the possible action space for current subgraphs."""
        subgraphs = self.subgraphs
        action_space = []
        for subgraph in subgraphs:
            parent_nodes = subgraph.ndata['_ID']
            parent_edges = set(subgraph.edata['_ID'].tolist())

            in_neighbors = self.graph.in_edges(parent_nodes, form="eid")
            out_neighbors = self.graph.out_edges(parent_nodes, form="eid")

            all_edges = set(torch.unique(torch.cat([in_neighbors, out_neighbors])).tolist())
            action = torch.tensor(list(all_edges - parent_edges))
            action_space.append(action)
        return action_space

    def update_state(self, action_space, action_probs):
        """Given the probabilities of each possible action, extend the current subgraphs."""
        subgraphs = []
        sample_actions = []
        sample_action_probs = []
        for idx in range(len(self.subgraphs)):
            subgraph = self.subgraphs[idx]
            action = action_space[idx]
            action_prob = action_probs[idx]
            sample_action_index = torch.multinomial(action_prob, action_prob.shape[0] // 3, replacement=False)
            sample_action_prob = action_prob[sample_action_index]
            sample_action = action[sample_action_index]

            sample_action_probs.append(sample_action_prob)
            sample_actions.append(sample_action)


raw_data_paths = {
        'train': os.path.join(main_dir, '../data/{}/{}.txt'.format("WN18RR_v1", "train")),
        'valid': os.path.join(main_dir, '../data/{}/{}.txt'.format("WN18RR_v1", "valid")),
    }

kg = KnowledgeGraph(raw_data_paths=raw_data_paths)
test_triples = kg.triplets["valid"][:64]

time1 = time.time()
kg.initialze_subgraphs(test_triples)
time2 = time.time()
action_space = kg.observe()
time3 = time.time()
action_probs = [torch.ones(action.shape) / action.shape[0] for action in action_space]

kg.update_state(action_space=action_space, action_probs=action_probs)
print(time2 - time1)
print(time3 - time2)
subgraph = kg.subgraphs[0]
print(subgraph.edata["type"])
print(subgraph.edata["_ID"])
print(kg.graph.edges("all"))
print(action_space[0])