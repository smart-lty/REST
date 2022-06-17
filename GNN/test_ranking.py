from operator import pos
import os
import argparse
import logging
import ipdb
from numpy.core.fromnumeric import size

import scipy.sparse as ssp
from tqdm import tqdm
import torch
import numpy as np
import dgl

from utils import *


def intialize_worker(model, adj_list, dgl_adj_list, params):
    global model_, adj_list_, dgl_adj_list_, params_, num_rels_
    model_, adj_list_, dgl_adj_list_, params_, num_rels_ = model, adj_list, dgl_adj_list, params, model.params.num_rels


global sizes_ 
sizes_ = []
def get_subgraphs(all_links, adj_list, dgl_adj_list):

    subgraphs = []
    labels = []

    for link in all_links:
        head, tail, rel = link[0], link[1], link[2]

        subgraph_nodes, _, subgraph_size = subgraph_extraction((head, tail), rel, adj_list, hop=params_.hop, enclosing_sub_graph=params.enclosing_sub_graph)
        sizes_.append(subgraph_size)

        subgraph = dgl_adj_list.subgraph(subgraph_nodes)

        heads, tails, eids = subgraph.edges('all')

        # indicator for edges between the head and tail
        indicator1 = torch.logical_and(heads==0, tails==1)
        # indicator for edges to be predicted
        indicator2 = torch.logical_and(subgraph.edata['type'] == rel, indicator1)

        if indicator2.sum() == 0:
            subgraph.add_edges(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(num_rels_).type(torch.LongTensor)
        else:
            subgraph.edata['type'][eids[indicator2]] = torch.tensor(num_rels_).type(torch.LongTensor)

        subgraphs.append(subgraph)
        labels.append(rel)

    batched_graph = dgl.batch(subgraphs)
    batched_labels = torch.LongTensor(labels)

    return (batched_graph, batched_labels)


def get_rel_rank(links, device):
    batch = get_subgraphs(links, adj_list_, dgl_adj_list_)
    graphs, labels = move_batch_to_device_dgl(batch, device)

    preds_rel = model_(graphs).detach().cpu()
    labels = labels.detach().cpu()

    rank = np.argwhere(np.argsort(-preds_rel)==labels.unsqueeze(-1))[1] + 1

    return rank.tolist()


def main(params):
    if params.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model = torch.load(params.model_path, map_location=device)
    model.device = device

    params.enclosing_sub_graph = model.params.enclosing_sub_graph
    adj_list, triplets, _, _, _, _ = process_files(params.file_paths, model.relation2id)

    if params.add_traspose_rels:
        adj_list_t = [adj.T for adj in adj_list]
        adj_list_aug = adj_list + adj_list_t
    else:
        adj_list_aug = adj_list

    dgl_adj_list = ssp_multigraph_to_dgl(adj_list_aug)

    ranks = []

    intialize_worker(model, adj_list, dgl_adj_list, params)

    cur_ = 0
    total = len(triplets['test'])
    while cur_ < total:
        next_ = min(cur_+params.batch_size, total)
        cur_batch = triplets['test'][cur_:next_]

        rank  = get_rel_rank(cur_batch, device)
        ranks += rank

        cur_ = next_

    isHit1List = [x for x in ranks if x <= 1]
    isHit3List = [x for x in ranks if x <= 3]
    isHit5List = [x for x in ranks if x <= 5]
    isHit10List = [x for x in ranks if x <= 10]
    hits_1 = len(isHit1List) / len(ranks)
    hits_3 = len(isHit3List) / len(ranks)
    hits_5 = len(isHit5List) / len(ranks)
    hits_10 = len(isHit10List) / len(ranks)

    mrr = np.mean(1 / np.array(ranks))

    logger.info(f'MRR | Hits@1 | Hits@3 | Hits@5 | Hits@10 : {mrr} | {hits_1} | {hits_3} |{hits_5} | {hits_10}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Testing script')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="fb_v2_margin_loss",
                        help="Experiment name. Log file with this name will be created")
    parser.add_argument("--dataset", "-d", type=str, default="FB237_v2",
                        help="Path to dataset")
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=False,
                        help='whether to only consider enclosing subgraph')
    parser.add_argument("--hop", type=int, default=2,
                        help="How many hops to go while extracting subgraphs?")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu id")
    parser.add_argument("--batch_size", "-bs", type=int, default=64,
                        help="test batch size")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='Whether to append adj matrix list with symmetric relations?')
    # parser.add_argument('--verbose', '-v', action='store_true', help='whether pring logs onto consoles')

    params = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)

    params.main_dir = os.path.relpath(os.path.dirname(os.path.abspath(__file__)))
    params.file_paths = {
        'train': os.path.join(params.main_dir, '../data', params.dataset, 'train.txt'),
        'test': os.path.join(params.main_dir, '../data', params.dataset, 'test.txt')
    }

    params.model_path = os.path.join(params.main_dir, 'experiments', params.experiment_name, 'best_graph_classifier.pth')

    logger = logging.getLogger()
    
    file_handler = logging.FileHandler(os.path.join(params.main_dir, 'experiments', params.experiment_name, 'log_test.txt'))
    logger.addHandler(file_handler)

    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    logger.info('============================================')

    main(params)
