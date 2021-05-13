import pickle
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import scipy
import sklearn

sns.set(color_codes=True)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
from torch import autograd

import sys

# my influence "package"
#from influence.influence_lib import get_influence_on_test_loss
from influence.influence_lib import *
from influence.hospital_training import Net, Trainer, BCELossDoubleBackward
from influence.utils import save, load
from influence.hospital_data import HospitalDataset

from config_my import NR_EPOCHS, DAMPING, TRAIN_DIR, MODEL_NAME, DATA_PATH

import time
from scipy.optimize import fmin_ncg
import cProfile
import os.path

from collections import defaultdict
from model.RankNet import *
from model.load_mslr import get_time
from model.metrics import NDCG
from model.utils import (
    eval_cross_entropy_loss,
    eval_ndcg_at_k,
    get_device,
    get_ckptdir,
    init_weights,
    load_train_vali_data,
    get_args_parser,
    save_to_ckpt,
)


np.random.seed(42)

# load dataset 
def load_data(standardize=True):
    data_fold = 'Fold1'
    data_dir = 'model/data/mslr-web30k/'
    if standardize and os.path.exists(data_dir+data_fold+'/standardized.pkl'):
        with open(data_dir+data_fold+'/standardized.pkl', 'rb') as fp:
            train_loader, df_train, valid_loader, df_valid, test_loader, df_test = pickle.load(fp)
    else:
        train_loader, df_train, valid_loader, df_valid = load_train_vali_data(data_fold, small_dataset=False)
        _, _, test_loader, df_test = load_train_vali_data(data_fold, small_dataset=True)

        if standardize:
            df_train, scaler = train_loader.train_scaler_and_transform()
            df_valid = valid_loader.apply_scaler(scaler)
            df_test = test_loader.apply_scaler(scaler)
            with open(data_dir+data_fold+'/standardized.pkl', 'wb') as fp:
                pickle.dump((train_loader, df_train, valid_loader, df_valid, test_loader, df_test), fp, pickle.HIGHEST_PROTOCOL)
    
    return train_loader, df_train, valid_loader, df_valid, test_loader, df_test


# load model with checkpoint
def get_model(train_loader, ckpt_epoch=50, train_algo=SUM_SESSION, double_precision=False, device='cuda:1'):
    net, net_inference, ckptfile = get_train_inference_net(
        train_algo, train_loader.num_features, ckpt_epoch, double_precision
    )
    net.to(device)
    net_inference.to(device)
    return net, net_inference


# eval & result
def eval_ndcg_at_k(inference_model, device, df_valid, valid_loader, k_list=[5, 10, 30], batch_size=1000000, phase="Eval"):
    # print("Eval Phase evaluate NDCG @ {}".format(k_list))
    ndcg_metrics = {k: NDCG(k) for k in k_list}
    qids, rels, scores = [], [], []
    inference_model.eval()
    with torch.no_grad():
        for qid, rel, x in valid_loader.generate_query_batch(df_valid, batch_size):
            if x is None or x.shape[0] == 0:
                continue
            y_tensor = inference_model.forward(torch.Tensor(x).to(device))
            scores.append(y_tensor.cpu().numpy().squeeze())
            qids.append(qid)
            rels.append(rel)

    qids = np.hstack(qids)
    rels = np.hstack(rels)
    scores = np.hstack(scores)
    result_df = pd.DataFrame({'qid': qids, 'rel': rels, 'score': scores})
    session_ndcgs = defaultdict(list)
    for qid in tqdm(result_df.qid.unique()):
        result_qid = result_df[result_df.qid == qid].sort_values('score', ascending=False)
        rel_rank = result_qid.rel.values
        for k, ndcg in ndcg_metrics.items():
            if ndcg.maxDCG(rel_rank) == 0:
                continue
            ndcg_k = ndcg.evaluate(rel_rank)
            if not np.isnan(ndcg_k):
                session_ndcgs[k].append(ndcg_k)

    ndcg_result = {k: np.mean(session_ndcgs[k]) for k in k_list}
    ndcg_result_print = ", ".join(["NDCG@{}: {:.5f}".format(k, ndcg_result[k]) for k in k_list])
    print(get_time(), "{} Phase evaluate {}".format(phase, ndcg_result_print))
    
    return ndcg_result, result_df


# 같은 query에 대한 모든 document pair loss를 반환
def get_prediction(X, Y, net, precision=torch.float32):
    if X is None or X.shape[0] == 0:
        return None, None, None, None
    Y = Y.reshape(-1, 1)
    rel_diff = Y - Y.T
    pos_pairs = (rel_diff > 0).astype(np.float32)
    num_pos_pairs = np.sum(pos_pairs, (0, 1))
    if num_pos_pairs == 0:
        return None, None, None, None
    if num_pos_pairs == 0:
        return None, None, None, None
    
    neg_pairs = (rel_diff < 0).astype(np.float32)
    num_pairs = 2 * num_pos_pairs  # num pos pairs and neg pairs are always the same

    pos_pairs = torch.tensor(pos_pairs, dtype=precision, device=device)
    neg_pairs = torch.tensor(neg_pairs, dtype=precision, device=device)

    X_tensor = torch.tensor(X, dtype=precision, device=device)
    y_pred = net(X_tensor)

    return X_tensor, y_pred, pos_pairs, neg_pairs
    
def criterion(y_pred, pos_pairs, neg_pairs, sigma=1.0):
    #training_algo == ACC_GRADIENT:
    l_pos = 1 + torch.exp(sigma * (y_pred - y_pred.t()))
    l_neg = 1 + torch.exp(- sigma * (y_pred - y_pred.t()))
    pos_loss = -sigma * pos_pairs / l_pos
    neg_loss = sigma * neg_pairs / l_neg
        
    return pos_loss, neg_loss