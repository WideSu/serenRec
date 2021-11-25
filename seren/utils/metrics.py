import torch

def get_metrics(rank_list, last, kpi):
    batch_size, topk = rank_list.size()
    expand_target = (last.squeeze()).unsqueeze(1).expand(-1, topk)
    hr = (rank_list == expand_target)
    
    ranks = (hr.nonzero(as_tuple=False)[:,-1] + 1).float()
    mrr = torch.reciprocal(ranks) # 1/ranks
    ndcg = 1/torch.log2(ranks + 1)

    metrics = {
        'hr': hr.sum(axis=1),
        'mrr': torch.cat([mrr, torch.zeros(batch_size - len(mrr))]),
        'ndcg': torch.cat([ndcg, torch.zeros(batch_size - len(ndcg))])
    }
    
    return metrics[kpi]

