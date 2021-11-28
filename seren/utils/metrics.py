import torch
import numpy as np

def accuracy_calculator(rank_list, last, kpis):
    batch_size, topk = rank_list.size()
    expand_target = (last.squeeze()).unsqueeze(1).expand(-1, topk)
    hr = (rank_list == expand_target)
    
    ranks = (hr.nonzero(as_tuple=False)[:,-1] + 1).float()
    mrr = torch.reciprocal(ranks) # 1/ranks
    ndcg = 1 / torch.log2(ranks + 1)

    metrics = {
        'hr': hr.sum(axis=1).double().mean().item(),
        'mrr': torch.cat([mrr, torch.zeros(batch_size - len(mrr))]).mean().item(),
        'ndcg': torch.cat([ndcg, torch.zeros(batch_size - len(ndcg))]).mean().item()
    }

    
    return [metrics[kpi] for kpi in kpis]

def diversity_calculator(rank_list, item_cate_matrix):
    rank_list = rank_list.long()
    ILD_perList = []
    for b in range(rank_list.size(0)):
        ILD = []
        for i in range(len(rank_list[b])):
            item_i_cate = item_cate_matrix[rank_list[b, i].item()]
            for j in range(i + 1, len(rank_list[b])):
                item_j_cate = item_cate_matrix[rank_list[b, j].item()]
                distance = np.linalg.norm(np.array(item_i_cate) - np.array(item_j_cate))
                ILD.append(distance)
        ILD_perList.append(np.mean(ILD))


    return torch.tensor(ILD_perList).mean().item()

