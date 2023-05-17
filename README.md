# Learning Rule-Induced Subgraph Representations for Inductive Relation Prediction

This is the official codebase of the paper Rule-Induced Subgraph Representations for Inductive Relation Prediction.


## File Tree
- data `(Save the dataset in this folder.)`
- GNN `(Core codes.)`
  - experiments `Save files for a experiment.`
  - manager `Trainer and Evaluator`
  - model  `Message function, Aggregate function, Update function and the whole model architecture.`
  - datasets.py  `Process data for training and evaluating.`
  - test_ranking.py `test MRR / hits@1 / hits@3 / hits@10 for a trained model.`
  - train.py `Training`
  - utils.py 
  - visualize.py `Visualization`
- README.md
- requirements.txt
- ablation_study.sh
- test_ranking.sh
- train.sh

## Setup

You can find the dependencies in `requirements.txt`. A script for installation is shown as follows:

```shell
conda create -n rest python=3.8
conda activate rest

pip install torch==2.0.0+cu117 torchvision==0.15.1+cu116 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install lmdb ipdb networkx scikit-learn scipy tqdm
```

## Train and Reproduction

To run REST, you can use the following command.
```shell
python GNN/train.py --gpu 0 -d WN18RR_v1 -e test --batch_size 512 -dim 32 --dropout 0 -l 6 --num_epochs 10
```

The configuration for all the results reported in the paper can be found in `GNN/experiments/`. To reproduce the results reported in the paper, just adjust the hyperparameters to the corresponding params. You can feel free to test other set of hyperparameters.

After train a model with experiment name `{exp_name}`, you can test it by running the following command:
```shell
python GNN/test_ranking.py --gpu 0 -d WN18RR_v1_ind -e {exp_name}
```

## Visualization
To visualize the learning rules by REST, you can run the following command:
```shell
python GNN/visualize.py
```