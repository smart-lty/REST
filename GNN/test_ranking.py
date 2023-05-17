import os
import argparse
import logging
import ipdb
from warnings import simplefilter
import torch
import time
import numpy as np
import dgl
import multiprocessing as mp
from utils import *
set_rand_seed(822)


def intialize_worker(model, adj_list, dgl_adj_list, device, params):
    global model_, adj_list_, dgl_adj_list_, device_, params_, num_rels_
    model_, adj_list_, dgl_adj_list_, device_, params_, num_rels_ = model, adj_list, dgl_adj_list, device, params, model.params.num_rels


global sizes_ 
sizes_ = []
def get_subgraphs(all_links, adj_list, dgl_adj_list, g_label, aug=False):

    subgraphs = []
    labels = []

    for link in all_links:
        head, tail, rel = link[0], link[1], link[2]
        if aug:
            subgraph_nodes, _, subgraph_size = aug_subgraph_extraction((head, tail), rel, adj_list, enclosing_hop=params_.hop, unclosing_hop=params_.un_hop)
        else:
            subgraph_nodes, _, subgraph_size = subgraph_extraction((head, tail), rel, adj_list, hop=params_.hop, enclosing_sub_graph=params.enclosing_sub_graph)
        sizes_.append(subgraph_size)

        subgraph = dgl_adj_list.subgraph(subgraph_nodes)

        heads, tails, eids = subgraph.edges('all')
        
        # indicator for edges between the head and tail
        indicator1 = torch.logical_and(heads==0, tails==1)
        # indicator for edges to be predicted
        indicator2 = torch.logical_and(subgraph.edata['type'] == rel, indicator1)

        if params_.add_traspose_rels:
            # indicator for the transpose relation of the target edge
            indicator3 = torch.logical_and(heads==1, tails==0)
            indicator4 = torch.logical_and(subgraph.edata['type'] == (rel + params_.num_rels // 2), indicator3)
            indicator2 = torch.logical_or(indicator4, indicator2)
        subgraph.edata["target_edge"] = torch.zeros(subgraph.edata['type'].shape).type(torch.BoolTensor)

        if indicator2.sum() == 0:
            subgraph.add_edge(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(rel).type(torch.LongTensor)
            subgraph.edata['target_edge'][-1] = torch.tensor(1).type(torch.BoolTensor)
            if params_.add_traspose_rels:
                subgraph.add_edge(1, 0)
                subgraph.edata['type'][-1] = torch.tensor(rel + params_.num_rels // 2).type(torch.LongTensor)
                subgraph.edata['target_edge'][-1] = torch.tensor(1).type(torch.BoolTensor)
        else:
            subgraph.edata['target_edge'][eids[indicator2]] = torch.ones(len(eids[indicator2])).type(torch.BoolTensor)

        subgraphs.append(subgraph)
        labels.append(g_label)

    batched_graph = dgl.batch(subgraphs)
    batched_labels = torch.LongTensor(labels)

    return batched_graph, batched_labels


def get_rel_rank(links):
    if len(links.shape) == 1:
        links = links[None, :]
    pos_links, neg_links = sample_neg(adj_list_, links, num_neg_samples_per_link=49)

    pos_graphs, pos_labels = get_subgraphs(pos_links, adj_list_, dgl_adj_list_, 1, aug=params_.aug)
    neg_graphs, neg_labels = get_subgraphs(neg_links, adj_list_, dgl_adj_list_, 0, aug=params_.aug)
    
    pos_graphs = send_graph_to_device(pos_graphs, device_)
    neg_graphs = send_graph_to_device(neg_graphs, device_)
    pos_scores = model_(pos_graphs).detach().cpu()
    neg_scores = model_(neg_graphs).reshape(-1, 49).detach().cpu()
    scores = torch.cat([pos_scores, neg_scores], dim=-1)

    rank = np.argwhere(np.argsort(-scores) == 0)[1] + 1

    return rank.tolist()


def main(params):
    simplefilter(action='ignore', category=UserWarning)

    if params.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model = torch.load(params.model_path, map_location=device)
    model.device = device
    model.params.device = device

    params.enclosing_sub_graph = model.params.enclosing_sub_graph
    if not hasattr(model.params, 'residual'):
        model.params.residual = params.residual
        
    adj_list, triplets, _, _, _, _ = process_files(params.file_paths, model.relation2id)

    if params.add_traspose_rels:
        adj_list_t = [adj.T for adj in adj_list]
        adj_list_aug = adj_list + adj_list_t
    else:
        adj_list_aug = adj_list

    params.num_rels = len(adj_list_aug)
    dgl_adj_list = ssp_multigraph_to_dgl(adj_list_aug)

    ranks = []

    start = time.time()
    
    if params.gpu < 0:
        with mp.Pool(processes=32, initializer=intialize_worker, initargs=(model, adj_list, dgl_adj_list, device, params)) as p:
            for rank in tqdm(p.imap(get_rel_rank, triplets['test']), total=len(triplets['test'])):
                ranks.append(rank)
    else:
        intialize_worker(model, adj_list, dgl_adj_list, device, params)
        cur_ = 0
        total = len(triplets['test'])
        with tqdm(total=total//params.batch_size) as pbar:
            while cur_ < total:
                next_ = min(cur_+params.batch_size, total)
                cur_batch = triplets['test'][cur_:next_]
                rank  = get_rel_rank(cur_batch)
                ranks += rank
                cur_ = next_
                pbar.update(1)
    
    # intialize_worker(model, adj_list, dgl_adj_list, device, params)
    # # first step: get all negative links
    # pos_links, neg_links = sample_neg(adj_list_, triplets['test'], num_neg_samples_per_link=49)
    
    # # second step: extract subgraphs for all positive and negative links via multi-processing
    # pos_graphs, neg_graphs = [], []
    
    # pool = mp.Pool(processes=32)
    # for pos_link in pos_links:
    #     pos_graphs.append(pool.apply_async(get_subgraphs, args=([pos_link], adj_list_, dgl_adj_list_, 1, params_.aug)))
    # for neg_link in neg_links:
    #     neg_graphs.append(pool.apply_async(get_subgraphs, args=([neg_link], adj_list_, dgl_adj_list_, 0, params_.aug)))
    # pool.close()
    # pool.join()
    # ipdb.set_trace()
        
                
    end = time.time()
    isHit1List = [x for x in ranks if x <= 1]
    isHit3List = [x for x in ranks if x <= 3]
    isHit5List = [x for x in ranks if x <= 5]
    isHit10List = [x for x in ranks if x <= 10]
    hits_1 = len(isHit1List) / len(ranks)
    hits_3 = len(isHit3List) / len(ranks)
    hits_5 = len(isHit5List) / len(ranks)
    hits_10 = len(isHit10List) / len(ranks)

    mrr = np.mean(1 / np.array(ranks))
    logger.info(f'Testing time used: {end - start} s')
    logger.info(f'MRR | Hits@1 | Hits@3 | Hits@5 | Hits@10 : {mrr} | {hits_1} | {hits_3} |{hits_5} | {hits_10}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Testing script')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="fb_v2_margin_loss",
                        help="Experiment name. Log file with this name will be created")
    parser.add_argument("--dataset", "-d", type=str, default="fb237_v2",
                        help="Path to dataset")
    parser.add_argument('--aug', type=bool, default=False,
                        help='whether to sample augmented subgraph')
    parser.add_argument("--hop", type=int, default=3,
                        help="How many hops to go while extracting subgraphs?")
    parser.add_argument("--un_hop", type=int, default=1,
                        help="How many hops to go while extracting augmented subgraphs? (only active when param 'aug' is True)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu id")
    parser.add_argument("--batch_size", "-bs", type=int, default=32,
                        help="test batch size")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=True,
                        help='Whether to append adj matrix list with symmetric relations?')
    parser.add_argument("--residual", "-res", type=bool, default=False)

    params = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    params.main_dir = os.path.relpath(os.path.dirname(os.path.abspath(__file__)))
    params.file_paths = {
        'train': os.path.join(params.main_dir, '../data', params.dataset, 'train.txt'),
        'test': os.path.join(params.main_dir, '../data', params.dataset, 'test.txt')
    }

    params.model_path = os.path.join(params.main_dir, 'experiments', params.experiment_name, 'best_graph_classifier.pth')

    logger = logging.getLogger()

    file_handler = logging.FileHandler(os.path.join(params.main_dir, 'experiments', params.experiment_name, 'log_test_ranking.txt'))
    logger.addHandler(file_handler)

    logger.info('============ Initialized logger ============')
    logger.info('\n'.join(f'{k}: {str(v)}' for k, v in sorted(dict(vars(params)).items())))

    logger.info('============================================')

    main(params)
    # while True:
    #     try:
    #         main(params)
    #         break
    #     except:
    #         params.batch_size = params.batch_size // 2
    #         logging.info(f"Run Time Error! Reduce batch size. Current batch size: {params.batch_size}")
