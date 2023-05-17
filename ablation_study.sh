# ablation of single-source initialization and full initialization
# nohup python GNN/train.py --gpu 5 -d WN18RR_v1 -e WN18RR_v1_fi --batch_size 512 -en True -dim 32 --dropout 0 -l 6 --num_epochs 10 > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 5 -d WN18RR_v2 -e WN18RR_v2_fi --batch_size 512 -en True -dim 32 --dropout 0 -l 5 --num_epochs 10 > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 5 -d WN18RR_v3 -e WN18RR_v3_fi --batch_size 128 -dim 32 --dropout 0.1 -l 6 --num_epochs 10 > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 5 -d WN18RR_v4 -e WN18RR_v4_fi --batch_size 256 -en True -dim 32 --dropout 0.1 -l 3 --num_epochs 10 > /dev/null 2>&1 &

# nohup python GNN/train.py --gpu 6 -d fb237_v1 -e fb237_v1_fi --batch_size 16 -dim 32 --dropout 0.1 -l 6 --num_epochs 10 > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 6 -d fb237_v2 -e fb237_v2_fi --batch_size 32 -dim 32 --dropout 0.1 -l 6 --num_epochs 10 -res True > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 5 -d fb237_v3 -e fb237_v3_fi --batch_size 12 -dim 32 --dropout 0.1 -l 6 --num_epochs 5 > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 7 -d fb237_v4 -e fb237_v4_fi --batch_size 4 -dim 32 --dropout 0.1 -l 5 --num_epochs 5 > /dev/null 2>&1 &

# nohup python GNN/train.py --gpu 7 -d nell_v1 -e nell_v1_fi --batch_size 128 -dim 16 --dropout 0.2 -l 6 --num_epochs 10 > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 7 -d nell_v2 -e nell_v2_fi --batch_size 32 -en True -dim 32 --dropout 0.1 -l 5 --num_epochs 10 > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 6 -d nell_v3 -e nell_v3_fi --batch_size 16 -en True -dim 16 --dropout 0.1 -l 6 --num_epochs 5 > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 5 -d nell_v4 -e nell_v4_fi --batch_size 32 -dim 32 --dropout 0.2 -l 4 --num_epochs 10 > /dev/null 2>&1 &

# ablation of message functions: GRU / ADD / MUL
# nohup python GNN/train.py --gpu 2 -d WN18RR_v1 -e WN18RR_v1_add --batch_size 512 -en True -dim 32 --dropout 0 -l 6 --num_epochs 10 --message add > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 3 -d WN18RR_v2 -e WN18RR_v2_add --batch_size 512 -en True -dim 32 --dropout 0 -l 5 --num_epochs 10 --message add > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 2 -d WN18RR_v3 -e WN18RR_v3_add --batch_size 128 -dim 32 --dropout 0.1 -l 6 --num_epochs 10 --message add > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 3 -d WN18RR_v4 -e WN18RR_v4_add --batch_size 256 -en True -dim 32 --dropout 0.1 -l 3 --num_epochs 10 --message add > /dev/null 2>&1 &

# nohup python GNN/train.py --gpu 2 -d fb237_v1 -e fb237_v1_add --batch_size 16 -dim 32 --dropout 0.1 -l 6 --num_epochs 10 --message add > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 6 -d fb237_v2 -e fb237_v2_add --batch_size 32 -dim 32 --dropout 0.1 -l 6 --num_epochs 10 -res True --message add > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 5 -d fb237_v3 -e fb237_v3_add --batch_size 12 -dim 32 --dropout 0.1 -l 6 --num_epochs 5 --message add > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 4 -d fb237_v4 -e fb237_v4_add --batch_size 4 -dim 32 --dropout 0.1 -l 5 --num_epochs 5 --message add > /dev/null 2>&1 &

# nohup python GNN/train.py --gpu 3 -d nell_v1 -e nell_v1_add --batch_size 128 -dim 16 --dropout 0.2 -l 6 --num_epochs 10 --message add > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 7 -d nell_v2 -e nell_v2_add --batch_size 32 -en True -dim 32 --dropout 0.1 -l 5 --num_epochs 10 --message add > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 6 -d nell_v3 -e nell_v3_add --batch_size 16 -en True -dim 16 --dropout 0.1 -l 6 --num_epochs 5 --message add > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 3 -d nell_v4 -e nell_v4_add --batch_size 32 -dim 32 --dropout 0.2 -l 4 --num_epochs 10 --message add > /dev/null 2>&1 &

# nohup python GNN/train.py --gpu 4 -d WN18RR_v1 -e WN18RR_v1_mul --batch_size 512 -en True -dim 32 --dropout 0 -l 6 --num_epochs 10 --message mul > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 5 -d WN18RR_v2 -e WN18RR_v2_mul --batch_size 512 -en True -dim 32 --dropout 0 -l 5 --num_epochs 10 --message mul > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 4 -d WN18RR_v3 -e WN18RR_v3_mul --batch_size 128 -dim 32 --dropout 0.1 -l 6 --num_epochs 10 --message mul > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 5 -d WN18RR_v4 -e WN18RR_v4_mul --batch_size 256 -en True -dim 32 --dropout 0.1 -l 3 --num_epochs 10 --message mul > /dev/null 2>&1 &

# nohup python GNN/train.py --gpu 4 -d fb237_v1 -e fb237_v1_mul --batch_size 16 -dim 32 --dropout 0.1 -l 6 --num_epochs 10 --message mul > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 6 -d fb237_v2 -e fb237_v2_mul --batch_size 32 -dim 32 --dropout 0.1 -l 6 --num_epochs 10 -res True --message mul > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 5 -d fb237_v3 -e fb237_v3_mul --batch_size 12 -dim 32 --dropout 0.1 -l 6 --num_epochs 5 --message mul > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 5 -d fb237_v4 -e fb237_v4_mul --batch_size 4 -dim 32 --dropout 0.1 -l 5 --num_epochs 5 --message mul > /dev/null 2>&1 &

# nohup python GNN/train.py --gpu 5 -d nell_v1 -e nell_v1_mul --batch_size 128 -dim 16 --dropout 0.2 -l 6 --num_epochs 10 --message mul > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 7 -d nell_v2 -e nell_v2_mul --batch_size 32 -en True -dim 32 --dropout 0.1 -l 5 --num_epochs 10 --message mul > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 6 -d nell_v3 -e nell_v3_mul --batch_size 16 -en True -dim 16 --dropout 0.1 -l 6 --num_epochs 5 --message mul > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 5 -d nell_v4 -e nell_v4_mul --batch_size 32 -dim 32 --dropout 0.2 -l 4 --num_epochs 10 --message mul > /dev/null 2>&1 &

# ablation of udate functions: LSTM / MLP
# nohup python GNN/train.py --gpu 6 -d WN18RR_v1 -e WN18RR_v1_mlp --batch_size 512 -en True -dim 32 --dropout 0 -l 6 --num_epochs 10 --update mlp > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 7 -d WN18RR_v2 -e WN18RR_v2_mlp --batch_size 512 -en True -dim 32 --dropout 0 -l 5 --num_epochs 10 --update mlp > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 6 -d WN18RR_v3 -e WN18RR_v3_mlp --batch_size 128 -dim 32 --dropout 0.1 -l 6 --num_epochs 10 --update mlp > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 7 -d WN18RR_v4 -e WN18RR_v4_mlp --batch_size 256 -en True -dim 32 --dropout 0.1 -l 3 --num_epochs 10 --update mlp > /dev/null 2>&1 &

# nohup python GNN/train.py --gpu 6 -d fb237_v1 -e fb237_v1_mlp --batch_size 16 -dim 32 --dropout 0.1 -l 6 --num_epochs 10 --update mlp > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 6 -d fb237_v2 -e fb237_v2_mlp --batch_size 32 -dim 32 --dropout 0.1 -l 6 --num_epochs 10 -res True --update mlp > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 7 -d fb237_v3 -e fb237_v3_mlp --batch_size 12 -dim 32 --dropout 0.1 -l 6 --num_epochs 5 --update mlp > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 6 -d fb237_v4 -e fb237_v4_mlp --batch_size 4 -dim 32 --dropout 0.1 -l 5 --num_epochs 5 --update mlp > /dev/null 2>&1 &

# nohup python GNN/train.py --gpu 7 -d nell_v1 -e nell_v1_mlp --batch_size 128 -dim 16 --dropout 0.2 -l 6 --num_epochs 10 --update mlp > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 7 -d nell_v2 -e nell_v2_mlp --batch_size 32 -en True -dim 32 --dropout 0.1 -l 5 --num_epochs 10 --update mlp > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 6 -d nell_v3 -e nell_v3_mlp --batch_size 16 -en True -dim 16 --dropout 0.1 -l 6 --num_epochs 5 --update mlp > /dev/null 2>&1 &
# nohup python GNN/train.py --gpu 7 -d nell_v4 -e nell_v4_mlp --batch_size 32 -dim 32 --dropout 0.2 -l 4 --num_epochs 10 --update mlp > /dev/null 2>&1 &