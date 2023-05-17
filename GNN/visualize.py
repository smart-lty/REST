import torch
import ipdb
import dgl
import copy
import itertools
import tqdm

# visualize on fb237_v4
model = torch.load(f"GNN/experiments/WN18RR_v1_ln_True_32_0_6_gru_lstm/best_graph_classifier.pth")

relation2id = model.relation2id
reverse_relation2id = {v: k for (k, v) in relation2id.items()}

num_rel = len(relation2id)

device = torch.device("cpu")

def generate_cycles(target_rel, relation_list, k):
    'generate all possible cycles with length k.'
    
    # create a cycle with length k
    src_nodes = torch.tensor([i for i in range(k)] + [i for i in range(1, k)] + [0])
    dst_nodes = torch.tensor([i for i in range(1, k)] + [0] + [i for i in range(k)])
    g = dgl.graph((src_nodes, dst_nodes))
    
    # generate all potential combinations
    all_edge_labels = [[target_rel] + list(item) for item in itertools.product(*[relation_list for i in range(1, k)])]
    
    graphs = []
    for edge_label in tqdm.tqdm(all_edge_labels):
        copy_g = copy.deepcopy(g)
        copy_g.edata["type"] = torch.tensor(edge_label + [i + len(relation_list) for i in edge_label])
        copy_g.edata["target_edge"] = torch.zeros(copy_g.edata['type'].shape).type(torch.BoolTensor)
        copy_g.edata["target_edge"][0] = torch.tensor(1).type(torch.BoolTensor)
        copy_g.edata["target_edge"][k] = torch.tensor(1).type(torch.BoolTensor)
        copy_g.ndata["degree"] = copy_g.out_degrees() + 1
        graphs.append(copy_g)
    
    return graphs

model.params.device = device
model = model.to(device)

if not hasattr(model.params, 'residual'):
        model.params.residual = False
        
for rel in range(num_rel):
    print("="*20)
    print(f"Generating cycles for rel {reverse_relation2id[rel]}...")
    graphs = generate_cycles(rel, list(range(num_rel)), 5)

    res = {}
    for graph in tqdm.tqdm(graphs):
        score = torch.sigmoid(model(graph)).squeeze().detach().cpu().numpy()
        # score = model(graph).squeeze().detach().cpu().numpy()
        edge_label = graph.edata["type"][:len(graph.edata["type"]) // 2].tolist()
        res[tuple(edge_label)] = score

    sorted_res = sorted(res.items(), key=lambda x:x[1])[::-1]
    for item in sorted_res[:5]:
        print("Rule head: ", end="")
        print(f"{reverse_relation2id[item[0][0]]} ", end="")
        print("Rule body: ", end="")
        print(f"{reverse_relation2id[item[0][1]]}", end="")
        for rel in item[0][2:]:
            print(f"-> {reverse_relation2id[rel]}", end="")
        print(f"   Score:{item[1]}")
    print("="*20)