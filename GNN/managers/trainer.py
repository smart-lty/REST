from re import I
import os
import logging
import time
import ipdb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics

class Trainer():
    def __init__(self, params, graph_classifier, train_data, valid_evaluator=None):
        self.graph_classifier = graph_classifier
        self.valid_evaluator = valid_evaluator
        self.params = params
        self.train_data = train_data

        self.updates_counter = 0

        model_params = list(self.graph_classifier.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_params), lr=params.lr, weight_decay=self.params.l2)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_params), lr=params.lr, weight_decay=self.params.l2)

        # self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='sum')
        if params.loss == "bce":
            self.criterion = nn.BCELoss()
        if params.loss == "mr":
            self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='sum')
        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = 0
        self.stop_training = False

    def train_epoch(self):
        total_loss = []
        all_labels = []
        all_scores = []

        dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        self.graph_classifier.train()
        model_params = list(self.graph_classifier.parameters())

        tic = time.time()
        for batch in dataloader:
            pos_graph, pos_label, neg_graph, neg_label = self.params.move_batch_to_device(batch, self.params.device)
            self.optimizer.zero_grad()
            time1 = time.time()
            score_pos = self.graph_classifier(pos_graph)
            score_neg = self.graph_classifier(neg_graph)
            score = torch.cat([score_pos, score_neg], dim=0).squeeze()
            label = torch.cat([pos_label, neg_label]).float()
            if self.params.loss == "bce":
                loss = self.criterion(torch.sigmoid(score), label)
            if self.params.loss == "mr":
                loss = self.criterion(score_pos.squeeze(1), score_neg.view(len(score_pos), -1).mean(dim=1), torch.Tensor([1]).to(device=self.params.device))
            # loss += self.params.l2 * sum(map(lambda x: torch.linalg.norm(x), model_params)) * 1.0 / len(model_params)
            loss.backward()

            total_loss.append(loss.item())

            self.optimizer.step()

            with torch.no_grad():
                all_scores += score_pos.squeeze(1).detach().cpu().tolist() + score_neg.squeeze(1).detach().cpu().tolist()
                all_labels += pos_label.tolist() + neg_label.tolist()

        auc = metrics.roc_auc_score(all_labels, all_scores)
        auc_pr = metrics.average_precision_score(all_labels, all_scores)
        acc = metrics.accuracy_score(y_true=all_labels, y_pred=[1 if i>=0.5 else 0 for i in all_scores])

        train_result = {'auc': auc, 'auc_pr': auc_pr, 'acc': acc}
        logging.info(f'Epoch {self.epoch} Training Performance:{train_result} in {str(time.time() - tic)} s ')


        if self.valid_evaluator:
            tic = time.time()
            result = self.valid_evaluator.eval()
            logging.info(f'Epoch {self.epoch} Validation Performance:{str(result)} in {str(time.time() - tic)} s ')


            res = result['auc_pr']
            if res >= self.best_metric:
                self.save_classifier()
                self.best_metric = res
                self.not_improved_count = 0
            else:
                self.not_improved_count += 1
                if self.not_improved_count > self.params.early_stop:
                    self.stop_training = True

        weight_norm = sum(map(lambda x: torch.norm(x), model_params)) * 1.0 / len(model_params)

        return np.mean(total_loss), weight_norm

    def train(self):
        self.reset_training_state()

        for epoch in range(1, self.params.num_epochs + 1):
            self.epoch = epoch
            time_start = time.time()
            loss, weight_norm = self.train_epoch()
            time_elapsed = time.time() - time_start
            
            logging.info(f'Epoch {epoch} with loss: {loss}, best validation AUC-PR: {self.best_metric}, weight_norm: {weight_norm} in {time_elapsed} s ')
            logging.info("=" * 100)

            if self.stop_training:
                logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
                break


    def save_classifier(self):
        torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'best_graph_classifier.pth'))  
        logging.info(f'Epoch {self.epoch} Better models found w.r.t AUC-PR. Saved it!')
