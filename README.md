# Code for AAAI 2023

## People
- Tianyu Liu `tianyuliu@miralab.ai`
- Zijie Geng `zijiegeng@miralab.ai`
- Jian Luo
- Yue shen

## Environment
1. `conda create -n rl_env python=3.7`
2. `conda activate rl_env`
3. `pip install -r requirements.txt` (Note: Some packages may not be available by using Pip Installation. Connect me with any problems.)

## Training

### GNN Training

For GNN training, please run `python GNN/train.py -d WN18RR_v1 --gpu 0 -e test`.
(To adjust more parameters, please refer to `train.py` line `54` to line `113`. Feel Free to ADJUST them!)

For GNN training with `nohup`, please run `nohup python GNN/train.py -d WN18RR_v1 --gpu 0 -e test > /dev/null 2>&1 &`.

#### Logs
If you have trained a model with experiment name `exp`, you can find logs in `GNN/experiments/{exp}` folder. It contains model file, train logs, and training parameters.

### RL Training
TBD.