import os
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import ipdb


class Evaluator():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data

    def eval(self, save=False):
        pos_scores = []
        pos_labels = []
        neg_scores = []
        neg_labels = []
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):

                pos_graph, pos_label, neg_graph, neg_label = self.params.move_batch_to_device(batch, self.params.device)
                
                score_pos = self.graph_classifier(pos_graph)
                score_neg = self.graph_classifier(neg_graph)

                # preds += torch.argmax(logits.detach().cpu(), dim=1).tolist()
                pos_scores += score_pos.squeeze(1).detach().cpu().tolist()
                neg_scores += score_neg.squeeze(1).detach().cpu().tolist()
                pos_labels += pos_label.tolist()
                neg_labels += neg_label.tolist()

        auc = metrics.roc_auc_score(pos_labels + neg_labels, pos_scores + neg_scores)
        auc_pr = metrics.average_precision_score(pos_labels + neg_labels, pos_scores + neg_scores)

        return {'auc': auc, 'auc_pr': auc_pr}