import os
import argparse
import logging
import json
from warnings import simplefilter

from scipy.sparse import SparseEfficiencyWarning
import torch

from datasets import *
from utils import *
from managers.evaluator import Evaluator
from managers.trainer import Trainer

## nohup python GNN/train.py -d WN18RR_v1 --gpu 0 -e test > /dev/null 2>&1 & ##

def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)
    
    params.db_path = os.path.join(params.main_dir, "..", f'data/{params.dataset}/subgraphs_enclose_{params.enclosing_sub_graph}_neg_{params.num_neg_samples_per_link}_hop_{params.hop}')

    if not os.path.isdir(params.db_path):
        generate_subgraph_datasets(params)
    
    train_data = SubgraphDataset(params.db_path, 'train', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels, 
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            dataset=params.dataset)
    valid_data = SubgraphDataset(params.db_path, 'valid', params.file_paths,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            dataset=params.dataset)
                            

    params.num_rels = train_data.num_rels
    params.inp_dim = params.emb_dim

    graph_classifier = initialize_model(params, params.load_model)

    logging.info(f"Device: {params.device}")
    logging.info(f"Input dim : {params.inp_dim}, # Relations : {params.num_rels}")

    valid_evaluator = Evaluator(params, graph_classifier, valid_data)

    trainer = Trainer(params, graph_classifier, train_data, valid_evaluator)

    logging.info('Starting training ...')

    trainer.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="default",
                        help="A folder with this name would be created to dump saved models and log files")
    parser.add_argument("--dataset", "-d", type=str,
                        help="Dataset string")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which GPU to use?")
    parser.add_argument('--load_model', action='store_true',
                        help='Load existing model?')
    # parser.add_argument('--verbose', '-v', action='store_true', help='whether print logs onto consoles')
    parser.add_argument("--train_file", "-tf", type=str, default="train",
                        help="Name of file containing training triplets")
    parser.add_argument("--valid_file", "-vf", type=str, default="valid",
                        help="Name of file containing validation triplets")

    # Training regime params
    parser.add_argument("--num_epochs", "-ne", type=int, default=100,
                        help="Learning rate of the optimizer")
    parser.add_argument("--early_stop", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--optimizer", type=str, default="Adam",
                        help="Which optimizer to use?")
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Learning rate of the optimizer")
    parser.add_argument("--l2", type=float, default=1e-7,
                        help="Regularization constant for GNN weights")
    parser.add_argument("--margin", type=float, default=10,
                        help="The margin between positive and negative samples in the max-margin loss")
    parser.add_argument("--using_jk", '-jk', type=bool, default=True,
                        help="whether to using jumping knowledge connection")

    # Data processing pipeline params
    parser.add_argument("--hop", type=int, default=2,
                        help="Enclosing subgraph hop number")
    parser.add_argument("--num_neg_samples_per_link", '-neg', type=int, default=1,
                        help="Number of negative examples to sample per positive link")
    parser.add_argument("--max_links", type=int, default=1000000,
                        help="Set maximum number of train links (to fit into memory)")
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0.0,
                        help='with what probability to sample constrained heads/tails while neg sampling')
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of dataloading processes")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='whether to append adj matrix list with symmetric relations')
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=False,
                        help='whether to only consider enclosing subgraph')

    # Model params
    parser.add_argument("--emb_dim", "-dim", type=int, default=64,
                        help="Entity embedding size")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout rate in GNN layers")
    parser.add_argument("--num_gcn_layers", "-l", type=int, default=3,
                        help="Number of GCN layers")
                    
    params = parser.parse_args()

    if params.experiment_name == "default":
        params.experiment_name == "_".join([params.dataset, "enclosing", params.enclosing_sub_graph, params.hop])

    logging.basicConfig(level=logging.INFO)

    initialize_experiment(params, __file__)

    params.file_paths = {
        'train': os.path.join(params.main_dir, '../data/{}/{}.txt'.format(params.dataset, params.train_file)),
        'valid': os.path.join(params.main_dir, '../data/{}/{}.txt'.format(params.dataset, params.valid_file))
    }
    os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu)

    if torch.cuda.is_available():
        params.device = torch.device('cuda')
        
    else:
        params.device = torch.device('cpu')

    params.collate_fn = collate_dgl
    params.move_batch_to_device = move_batch_to_device_dgl

    print(params)
    main(params)
