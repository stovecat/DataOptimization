from sklearn.metrics import average_precision_score
import numpy as np

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer
import pickle

def dcg_score(y_true, y_score, k=5):
    """Discounted cumulative gain (DCG) at rank K.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array, shape = [n_samples, n_classes]
        Predicted scores.
    k : int
        Rank.

    Returns
    -------
    score : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)

def ndcg_score(ground_truth, predictions, k=5):
    actual = dcg_score(ground_truth, predictions, k)
    best = dcg_score(ground_truth, ground_truth, k)
    score = float(actual) / float(best)
    return score

def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)
def average_precision(r):
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)
def MAP(prediction, label, min_pos_label):
    label = np.array([1.0 if l >= min_pos_label else 0. for l in label])
    rs = []
    for pred_list_score, pred_list, in zip(prediction, label):
        pred_url_score = zip(pred_list, pred_list_score)
        pred_url_score = sorted(pred_url_score, key=lambda x: x[1], reverse=True)

        r = [0.0] * len(pred_list_score)
        for i in range(0, len(pred_list_score)):
            (url, score) = pred_url_score[i]
            if url == 1.0:
                r[i] = 1.0
        rs.append(r)

    return np.mean([average_precision(r) for r in rs])

def evaluation(prediction, label, min_pos_label=1.):
    for_analysis = []
    metric = {'P@1': 0., 'P@3': 0., 'P@5': 0., 'P@10': 0., 'P@30': 0., 'MAP': 0., 
                  'NDCG@3': 0., 'NDCG@5': 0., 'NDCG@10': 0., 'MRR': 0.}
    no_answer = 0
    for idx, (p, l) in enumerate(zip(prediction, label)):
        assert len(p) == len(l)
        merged = [(i, _p, _l) for i, (_p, _l) in enumerate(zip(p, l))]
        sorted_rank = sorted(merged, key=lambda m: m[1], reverse=True)
        #print(sorted_rank)
#         for sr in sorted_rank:
#             print("Rank: "+str(sr[0])+"\tPred: "+str(sr[1])+"\tLabel: "+str(sr[2]))
#         print()
        
        rank = []
        for i, m in enumerate(sorted_rank):
            if m[2] >= min_pos_label:
                rank.append(i+1)
        #for r in rank:
        #    print("Rank: "+str(r))
        #print(sum([m[2] for m in sorted_rank[:3]]) / 3.0)
        if len(rank) > 0:
            metric['MRR'] += 1 / float(rank[0])
            metric['MAP'] += average_precision_score(np.array([1.0 if _l >= min_pos_label else 0. for _l in l]), np.array(p))
            l_index = np.array(l)
            p_divide = [3., 5., 10., 30.]
#             for p_idx in range(len(p_divide)):
#                 if len(rank) < p_divide[p_idx]:
#                     p_divide[p_idx] = len(rank)
            metric['P@1'] += sum([1.0 if m[2] >= min_pos_label else 0. for m in sorted_rank[:1]])
            metric['P@3'] += sum([1.0 if m[2] >= min_pos_label else 0. for m in sorted_rank[:3]]) / p_divide[0]
            metric['P@5'] += sum([1.0 if m[2] >= min_pos_label else 0. for m in sorted_rank[:5]]) / p_divide[1]
            metric['P@10'] += sum([1.0 if m[2] >= min_pos_label else 0. for m in sorted_rank[:10]]) / p_divide[2]
            metric['P@30'] += sum([1.0 if m[2] >= min_pos_label else 0. for m in sorted_rank[:30]]) / p_divide[3]
#             metric['NDCG@3'] += ndcg_score(l_index, np.array(p), k=3)
#             metric['NDCG@5'] += ndcg_score(l_index, np.array(p), k=5)            
#             metric['NDCG@10'] += ndcg_score(l_index, np.array(p), k=10)
#             print(idx)
#             print(l_index)
#             print(p)
#             print([m[2] for m in sorted_rank[:5]])
#             print(ndcg_score(l_index, np.array(p), k=5))
#             print()
            #print(sum([m[2] for m in sorted_rank[:5]]) / 5.0)
            #for_analysis.append(ndcg_score(l_index, np.array(p), k=3))
        else:
            no_answer += 1
#     with open('analysis.pkl', 'w') as fp:
#         pickle.dump(for_analysis, fp, pickle.HIGHEST_PROTOCOL)
    #print(metric['NDCG@5'])
    #print("no_answer: "+str(no_answer))
    #print(len(label), no_answer)
#     if (len(label) - no_answer) == 0:
#         print(len(label))
#         print(sum(sum(label)))
#         print(prediction)
#         assert (len(label) - no_answer) != 0
    #print(no_answer)
    for k in metric.keys():
        metric[k] /= (len(label) - no_answer)
    #print(sum(rr), cnt, sum(rr)/cnt)    
    #print(MAP(prediction, label))
    return metric

def print_metric(metric):
    for k in sorted(metric.keys()):
        if k == 'global_step':
            print(k+':\t'+str(int(metric[k])))
        else:
            print(k+':\t'+str(round(metric[k], 4)))
    print("")

