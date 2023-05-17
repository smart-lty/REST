import contextlib
import logging
from os import link
from tqdm import tqdm
import lmdb
import random
import torch
import dgl
import pickle
import multiprocessing as mp
import numpy as np
import scipy.sparse as ssp
import networkx as nx
import os
import json
from model.CGNN import CycleGNN

# ================================= Randomize Related Utils ==============================================
def set_rand_seed(seed=1):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False       
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


# ================================= Subgraph Related Utils ==============================================

# save to database
def links2subgraphs(A, graphs, params, max_label_value=None):
    """
    extract enclosing subgraphs, write map mode + named dbs
    """
    subgraph_sizes = []
    BYTES_PER_DATUM = get_average_subgraph_size(100, list(graphs.values())[0]['pos'], A, params) * 1.5

    links_length = 0
    for split_name, split in graphs.items():
        links_length += (len(split['pos']) + len(split['neg'])) * 2
    map_size = links_length * BYTES_PER_DATUM
    env = lmdb.open(params.db_path, map_size=map_size, max_dbs=6)
    def extraction_helper(A, links, g_labels, split_env):
        with env.begin(write=True, db=split_env) as txn:
            txn.put('num_graphs'.encode(), len(links).to_bytes(int.bit_length(len(links)), byteorder='little'))

        with mp.Pool(processes=None, initializer=intialize_worker, initargs=(A, params, max_label_value)) as p:
            args_ = zip(range(len(links)), links, g_labels)
            for str_id, datum in tqdm(p.imap(extract_save_subgraph, args_), total=len(links)):
                subgraph_sizes.append(datum['subgraph_size'])
                with env.begin(write=True, db=split_env) as txn:
                    txn.put(str_id, serialize(datum))

    for split_name, split in graphs.items():
        logging.info(f"Extracting enclosing subgraphs for positive links in {split_name} set")

        labels = np.ones(len(split['pos']))
        db_name_pos = f'{split_name}_pos'
        split_env = env.open_db(db_name_pos.encode())
        extraction_helper(A, split['pos'], labels, split_env)
        logging.info(f"Extracting enclosing subgraphs for negative links in {split_name} set")

        labels = np.zeros(len(split['neg']))
        db_name_neg = f'{split_name}_neg'
        split_env = env.open_db(db_name_neg.encode())
        extraction_helper(A, split['neg'], labels, split_env)


def sample_neg(adj_list, edges, num_neg_samples_per_link=1, max_size=1000000, constrained_neg_prob=0):
    pos_edges = edges
    neg_edges = []

    # if max_size is set, randomly sample train links
    if max_size < len(pos_edges):
        perm = np.random.permutation(len(pos_edges))[:max_size]
        pos_edges = pos_edges[perm]

    # sample negative links for train/test
    n, r = adj_list[0].shape[0], len(adj_list)

    # possible head and tails for each relation
    valid_heads = [adj.tocoo().row.tolist() for adj in adj_list]
    valid_tails = [adj.tocoo().col.tolist() for adj in adj_list]

    neg_n = 0
    while len(neg_edges) < num_neg_samples_per_link * len(pos_edges):
        neg_head, neg_tail, rel = pos_edges[neg_n % len(pos_edges)][0], pos_edges[neg_n % len(pos_edges)][1], pos_edges[neg_n % len(pos_edges)][2]
        
        if np.random.uniform() < constrained_neg_prob:
            if np.random.uniform() < 0.5:
                neg_head = np.random.choice(valid_heads[rel])
            else:
                neg_tail = np.random.choice(valid_tails[rel])
        else:
            if np.random.uniform() < 0.5:
                neg_head = np.random.choice(n)
            else:
                neg_tail = np.random.choice(n)

        if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
            neg_edges.append([neg_head, neg_tail, rel])
            neg_n += 1

    neg_edges = np.array(neg_edges)
    return pos_edges, neg_edges


def get_average_subgraph_size(sample_size, links, A, params):
    total_size = 0
    for (n1, n2, r_label) in links[np.random.choice(len(links), sample_size)]:
        if params.aug:
            subgraph_nodes, subgraph_labels, subgraph_size = aug_subgraph_extraction((n1, n2), r_label, A, params.hop, params.un_hop)
        else:
            subgraph_nodes, subgraph_labels, subgraph_size = subgraph_extraction((n1, n2), r_label, A, params.hop, params.enclosing_sub_graph)
        datum = {'subgraph_nodes': subgraph_nodes, 'subgraph_labels': subgraph_labels, 'subgraph_size': subgraph_size}
        total_size += len(serialize(datum))
    return total_size / sample_size


def intialize_worker(A, params, max_label_value):
    global A_, params_, max_label_value_
    A_, params_, max_label_value_ = A, params, max_label_value


def extract_save_subgraph(args_):
    idx, (n1, n2, r_label), g_labels = args_
    if params_.aug:
        subgraph_nodes, subgraph_labels, subgraph_size = aug_subgraph_extraction((n1, n2), r_label, A_, params_.hop, params_.un_hop)
    else:
        subgraph_nodes, subgraph_labels, subgraph_size = subgraph_extraction((n1, n2), r_label, A_, params_.hop, params_.enclosing_sub_graph)
    datum = {'subgraph_nodes': subgraph_nodes, 'subgraph_labels': subgraph_labels, 'subgraph_size': subgraph_size}

    # padding the length to be 8
    str_id = '{:08}'.format(idx).encode('ascii')

    return (str_id, datum)


def aug_subgraph_extraction(ind, rel, A_list, enclosing_hop=3, unclosing_hop=1):
    # bidirectional
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T

    # type: set
    root1_en = get_neighbor_nodes({ind[0]}, A_incidence, enclosing_hop)
    root2_en= get_neighbor_nodes({ind[1]}, A_incidence, enclosing_hop)

    # k1 hop enclosing subgraph
    subgraph_nei_nodes_int = root1_en.intersection(root2_en)
    
    if len(subgraph_nei_nodes_int) == 0:
        root1_en = get_neighbor_nodes({ind[0]}, A_incidence, enclosing_hop * 2)
        root2_en= get_neighbor_nodes({ind[1]}, A_incidence, enclosing_hop * 2)
        # k1 hop enclosing subgraph
        subgraph_nei_nodes_int = root1_en.intersection(root2_en)
        root1_un = get_neighbor_nodes({ind[0]}, A_incidence, unclosing_hop)
        root2_un= get_neighbor_nodes({ind[1]}, A_incidence, unclosing_hop)
        
        # k2 hop unclosing subgraph
        subgraph_nei_nodes_un = root1_un.union(root2_un)
        
        subgraph_nodes = subgraph_nei_nodes_int.union(subgraph_nei_nodes_un)
        subgraph_nodes = list(ind) + [i for i in subgraph_nodes if i not in ind]

    else:
        subgraph_nodes = list(ind) + [i for i in subgraph_nei_nodes_int if i not in ind]
    
    subgraph_label = rel
    
    subgraph_size = len(subgraph_nodes)

    assert len(subgraph_nodes) == len(set(subgraph_nodes))

    return subgraph_nodes, subgraph_label, subgraph_size


def subgraph_extraction(ind, rel, A_list, hop=1, enclosing_sub_graph=False):
    """
    extract the h-hop enclosing subgraphs around link 'ind'

    ind: (head, tail)
    A_list: the list of adjacency matrix each corresponding to each relation; built from training data
    """

    # csc matrices
    # bidirectional
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T

    # type: set
    root1_neigh = get_neighbor_nodes({ind[0]}, A_incidence, hop)
    root2_neigh = get_neighbor_nodes({ind[1]}, A_incidence, hop)

    subgraph_nei_nodes_int = root1_neigh.intersection(root2_neigh)
    subgraph_nei_nodes_un = root1_neigh.union(root2_neigh)

    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    # ! In the code of grail, actually, the extracted non-enclosing subgraph is not correct, as a lot of repeat nodes are included.
    if enclosing_sub_graph:
        subgraph_nodes = list(ind) + [i for i in subgraph_nei_nodes_int if i not in ind]
    else:
        subgraph_nodes = list(ind) + [i for i in subgraph_nei_nodes_un if i not in ind]

    subgraph_label = rel

    subgraph_size = len(subgraph_nodes)

    assert len(subgraph_nodes) == len(set(subgraph_nodes))

    return subgraph_nodes, subgraph_label, subgraph_size


def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = []
    for _ in range(h):
        with contextlib.suppress(StopIteration):
            lvls.append(next(bfs_generator))
    return set().union(*lvls)


def _bfs_relational(adj, roots, max_nodes_per_hop=None):
    """
    BFS for graphs.
    Modified from dgl.contrib.data.knowledge_graph to accomodate node sampling
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(next_lvl, max_nodes_per_hop))

        yield next_lvl

        current_lvl = set.union(next_lvl)


def _get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors.
    Directly copied from dgl.contrib.data.knowledge_graph"""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    return set(ssp.find(sp_neighbors)[1])


def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def serialize(data):
    data_tuple = tuple(data.values())
    return pickle.dumps(data_tuple)


def deserialize(data):
    data_tuple = pickle.loads(data)
    keys = ('subgraph_nodes', 'subgraph_labels', 'subgraph_size')
    return dict(zip(keys, data_tuple))


def get_edge_count(adj_list):
    count = []
    for adj in adj_list:
        count.append(len(adj.tocoo().row.tolist()))
    return np.array(count)


# ================================= Subgraph Related Utils ==============================================

# ================================= Graph Related Utils ==============================================

def incidence_matrix(adj_list):
    '''
    adj_list: List of sparse adjacency matrices
    '''
    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)


def ssp_multigraph_to_dgl(graph, n_feats=None):
    """
    Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
    """

    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    # Add edges
    for rel, adj in enumerate(graph):
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):

            nx_triplets.append((src, dst, {'type': rel}))
        g_nx.add_edges_from(nx_triplets)
    # make dgl graph
    g_dgl = dgl.from_networkx(g_nx, edge_attrs=['type'])
    # add node features
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)

    return g_dgl


def collate_dgl(samples):
    # The input `samples` is a list of pairs
    pos_subgraph, pos_label, neg_subgraphs, neg_labels = map(list, zip(*samples))
    for i, subgraph in enumerate(pos_subgraph):
        subgraph.edata["batch_id"] = torch.ones(subgraph.edata["type"].shape[0], dtype=torch.long) * i
    batched_pos_subgraph = dgl.batch(pos_subgraph)
    batched_pos_label = pos_label

    neg_subgraphs = [item for sublist in neg_subgraphs for item in sublist]
    neg_labels = [item for sublist in neg_labels for item in sublist]

    for i, subgraph in enumerate(neg_subgraphs):
        subgraph.edata["batch_id"] = torch.ones(subgraph.edata["type"].shape[0], dtype=torch.long) * i
    
    batched_neg_subgraphs = dgl.batch(neg_subgraphs)
    batched_neg_labels = neg_labels

    return batched_pos_subgraph, batched_pos_label, batched_neg_subgraphs, batched_neg_labels


def move_batch_to_device_dgl(batch, device):
    batched_pos_subgraph, batched_pos_label, batched_neg_subgraphs, batched_neg_labels = batch

    g_dgl_pos = send_graph_to_device(batched_pos_subgraph, device)
    g_dgl_neg = send_graph_to_device(batched_neg_subgraphs, device)

    batched_pos_label = torch.LongTensor(batched_pos_label).to(device=device)
    batched_neg_labels = torch.LongTensor(batched_neg_labels).to(device=device)

    return g_dgl_pos, batched_pos_label, g_dgl_neg, batched_neg_labels


def send_graph_to_device(g, device):
    # nodes
    g = g.to(device)
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).to(device)

    # edges
    labels = g.edge_attr_schemes()
    for l in labels.keys():
        g.edata[l] = g.edata.pop(l).to(device)
    return g


# ================================= Graph Related Utils ==============================================

# ================================= Data Related Utils ==============================================


def process_files(files, saved_relation2id=None):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id

    triplets = {}

    ent = 0
    rel = 0
    for file_type, file_path in files.items():
        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1
            if not saved_relation2id and triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel += 1

            # Save the triplets corresponding to only the known relations
            if triplet[1] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to each relation. 
    # NOTE: constructed only from the train data.
    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(ssp.csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))

    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation


def save_to_file(directory, file_name, triplets, id2entity, id2relation):
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w") as f:
        for s, o, r in triplets:
            f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')

# ================================= Data Related Utils ==============================================

# ================================= Initialization Related Utils ==============================================


def initialize_experiment(params, file_name):
    """
    Makes the experiment directory, sets standard paths and initializes the logger
    """
    params.main_dir = os.path.relpath(os.path.dirname(os.path.abspath(__file__)))
    exps_dir = os.path.join(params.main_dir, 'experiments')
    if not os.path.exists(exps_dir):
        os.makedirs(exps_dir)
    params.exp_dir = os.path.join(exps_dir, params.experiment_name)
    if not os.path.exists(params.exp_dir):
        os.makedirs(params.exp_dir)
    if file_name.endswith('test_auc.py'):
        file_handler = logging.FileHandler(os.path.join(params.exp_dir, "log_test_auc.txt"))

    elif file_name.endswith('test_ranking.py'):
        file_handler = logging.FileHandler(os.path.join(params.exp_dir, "log_test_ranking.txt"))

    else:
        file_handler = logging.FileHandler(os.path.join(params.exp_dir, "log_train.txt"))

    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join(f'{k}: {str(v)}' for k, v in sorted(dict(vars(params)).items())))

    logger.info('============================================')
    with open(os.path.join(params.exp_dir, "params.json"), 'w') as fout:
        json.dump(vars(params), fout)


def initialize_model(params, load_model=False):
    '''
    relation2id: the relation to id mapping, this is stored in the model and used when testing
    model: the type of model to initialize/load
    load_model: flag which decide to initialize the model or load a saved model
    '''

    if load_model and os.path.exists(os.path.join(params.exp_dir, 'best_graph_classifier.pth')):
        logging.info(f"Loading existing model from {os.path.join(params.exp_dir, 'best_graph_classifier.pth')}")

        graph_classifier = torch.load(os.path.join(params.exp_dir, 'best_graph_classifier.pth'), map_location=params.device).to(device=params.device)
    else:
        relation2id_path = os.path.join(params.main_dir, f'../data/{params.dataset}/relation2id.json')
        with open(relation2id_path) as f:
            relation2id = json.load(f)

        logging.info('No existing model found. Initializing new model..')
        graph_classifier = CycleGNN(params, relation2id).to(device=params.device)

    return graph_classifier

# ================================= Initialization Related Utils ==============================================

