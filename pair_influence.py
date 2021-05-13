import random
import pickle
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
#from tqdm.notebook import tqdm
from tqdm import tqdm

import scipy
import sklearn

sns.set(color_codes=True)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
from torch import autograd

import sys

# try:
#     from apex.parallel import DistributedDataParallel as DDP
#     from apex.fp16_utils import *
#     from apex import amp, optimizers
#     from apex.multi_tensor_apply import multi_tensor_applier
# except ImportError:
#     raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

# my influence "package"
#from influence.influence_lib import get_influence_on_test_loss
#from influence.influence_lib import params_to_list
#from influence.utils import save, load

#from config_my import NR_EPOCHS, DAMPING, TRAIN_DIR, MODEL_NAME, DATA_PATH

import time
from scipy.optimize import fmin_ncg
import cProfile
import os.path

from collections import defaultdict
from model.RankNet import *
from model.load_mslr import get_time, NaverLoader, MQ2008semiLoader, NaverClickLoader
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

USE_AMP = False

def save(model, path):
    try:
        torch.save(model.module.state_dict(), path)
    except AttributeError:
        torch.save(model.state_dict(), path)


def load(ModelClass, path, **kwargs):
    model = ModelClass(**kwargs)
    model.load_state_dict(torch.load(path))
    return model


# load dataset 
def load_naver_data(drop_high_rel=False):
    train_loader = NaverLoader(data_type='train', drop_high_rel=drop_high_rel)
    valid_loader = NaverLoader(data_type='valid', drop_high_rel=drop_high_rel)
    test_loader = NaverLoader(data_type='test', drop_high_rel=drop_high_rel)
    return train_loader, train_loader.df, valid_loader, valid_loader.df, test_loader, test_loader.df


def load_mq2008semi_data(device):
    train_loader = MQ2008semiLoader(data_type='train', device=device)
    valid_loader = MQ2008semiLoader(data_type='vali', device=device)
    test_loader = MQ2008semiLoader(data_type='test', device=device)
    return train_loader, train_loader.df, valid_loader, valid_loader.df, test_loader, test_loader.df


def load_naver_click_data(device):
    train_loader = NaverClickLoader(data_type='train', device=device)
    valid_loader = NaverClickLoader(data_type='valid', device=device)
    test_loader = NaverClickLoader(data_type='test', device=device)
    return train_loader, train_loader.df, valid_loader, valid_loader.df, test_loader, test_loader.df


def load_data(standardize=True, device=1, dataset_type='mslr-web30k', drop_high_rel=False):
    if dataset_type in ['mslr-web30k', 'mslr-web10k']:
        data_fold = 'Fold1'
        data_dir = 'model/data/'+dataset_type+'/'
        pkl_name = '/standardized.pkl'
        if device == 0:
            pkl_name = '/standardized_cuda0.pkl'
        if standardize and os.path.exists(data_dir+data_fold+pkl_name):
            with open(data_dir+data_fold+pkl_name, 'rb') as fp:
                train_loader, df_train, valid_loader, df_valid, test_loader, df_test = pickle.load(fp)
        else:
            train_loader, df_train, valid_loader, df_valid = load_train_vali_data(data_fold, small_dataset=False, 
                                                                                  data_type=dataset_type)
            _, _, test_loader, df_test = load_train_vali_data(data_fold, small_dataset=True, data_type=dataset_type)

            if standardize:
                df_train, scaler = train_loader.train_scaler_and_transform()
                df_valid = valid_loader.apply_scaler(scaler)
                df_test = test_loader.apply_scaler(scaler)
                with open(data_dir+data_fold+pkl_name, 'wb') as fp:
                    pickle.dump((train_loader, df_train, valid_loader, df_valid, test_loader, df_test), fp, pickle.HIGHEST_PROTOCOL)
    elif dataset_type == 'naver':
        data_fold = ''
        data_dir = 'model/data/naver/'
        if drop_high_rel:
            train_loader, df_train, valid_loader, df_valid, test_loader, df_test = load_naver_data(drop_high_rel)
        else:
            pkl_name = '/cuda1.pkl'
            if device == 0:
                pkl_name = '/cuda0.pkl'
            if os.path.exists(data_dir+data_fold+pkl_name):
                with open(data_dir+data_fold+pkl_name, 'rb') as fp:
                    train_loader, df_train, valid_loader, df_valid, test_loader, df_test = pickle.load(fp)
            else:
                train_loader, df_train, valid_loader, df_valid, test_loader, df_test = load_naver_data()
                with open(data_dir+data_fold+pkl_name, 'wb') as fp:
                    pickle.dump((train_loader, df_train, valid_loader, df_valid, test_loader, df_test), 
                                fp, pickle.HIGHEST_PROTOCOL)
    elif dataset_type == 'mq2008-semi':
        data_fold = ''
        data_dir = 'model/data/MQ2008-semi/'
        pkl_name = '/cuda1.pkl'
        if device == 0:
            pkl_name = '/cuda0.pkl'
        if os.path.exists(data_dir+data_fold+pkl_name):
            with open(data_dir+data_fold+pkl_name, 'rb') as fp:
                train_loader, df_train, valid_loader, df_valid, test_loader, df_test = pickle.load(fp)
        else:
            train_loader, df_train, valid_loader, df_valid, test_loader, df_test = load_mq2008semi_data(device)
            with open(data_dir+data_fold+pkl_name, 'wb') as fp:
                pickle.dump((train_loader, df_train, valid_loader, df_valid, test_loader, df_test), 
                            fp, pickle.HIGHEST_PROTOCOL)
    elif dataset_type == 'naver_click':
        data_fold = ''
        data_dir = 'model/data/naver_click/'
        pkl_name = '/cuda1.pkl'
        if device == 0:
            pkl_name = '/cuda0.pkl'
        if os.path.exists(data_dir+data_fold+pkl_name):
            with open(data_dir+data_fold+pkl_name, 'rb') as fp:
                train_loader, df_train, valid_loader, df_valid, test_loader, df_test = pickle.load(fp)
        else:
            train_loader, df_train, valid_loader, df_valid, test_loader, df_test = load_naver_click_data(device)
            with open(data_dir+data_fold+pkl_name, 'wb') as fp:
                pickle.dump((train_loader, df_train, valid_loader, df_valid, test_loader, df_test), 
                            fp, pickle.HIGHEST_PROTOCOL)    
    else:
        raise NotImplementedError
        
        
    return train_loader, df_train, valid_loader, df_valid, test_loader, df_test


args = {}
args["start_epoch"] = 0
args['additional_epoch'] = 50
args['lr'] = 0.01
args['optim'] = 'adam'
args['train_algo'] = SUM_SESSION
args['double_precision'] = False
args['standardize'] = True
args['small_dataset'] = False
args['debug'] = False#True
args['output_dir'] = '/model/ranknet/ranking_output/'


def train_rank_net(
    train_loader, valid_loader, df_valid,
    start_epoch=0, additional_epoch=100, lr=0.0001, optim="adam",
    train_algo=SUM_SESSION,
    double_precision=False, standardize=False,
    small_dataset=False, debug=False,
    output_dir="/tmp/ranking_output/",
    opt=None,
    log=True,
    device=0,
    seed=7777):
    """

    :param start_epoch: int
    :param additional_epoch: int
    :param lr: float
    :param optim: str
    :param train_algo: str
    :param double_precision: boolean
    :param standardize: boolean
    :param small_dataset: boolean
    :param debug: boolean
    :return:
    """
    
    print("start_epoch:{}, additional_epoch:{}, lr:{}".format(start_epoch, additional_epoch, lr))
    writer = SummaryWriter(output_dir)

    precision = torch.float64 if double_precision else torch.float32

    # get training and validation data:
    data_fold = 'Fold1'

    net, _, ckptfile = get_train_inference_net(
        train_algo, train_loader.num_features, start_epoch, double_precision, opt, log
    )    
    net.cuda(device)
    net_inference = net
    torch.backends.cudnn.benchmark=False
    
    # initialize to make training faster
    clear_seed_all(seed)
    net.apply(init_weights)
    if train_loader.dataset_type == 'naver':
        lr = 1e-2
        wd = 0.
    elif train_loader.dataset_type == 'mq2008-semi':
        lr = 5e-3
        wd = 0.
    elif train_loader.dataset_type == 'naver_click':
        lr = 1e-2
        wd = 0.
    else:
        lr = 1e-2
        wd = 0.
    if optim == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    elif optim == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError("Optimization method {} not implemented".format(optim))
    print(optimizer)
    
#     if USE_AMP:
#         net, optimizer = amp.initialize(net, optimizer)


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)

    loss_func = None
    if train_algo == BASELINE:
        loss_func = torch.nn.BCELoss()
        loss_func.cuda()

    losses = []

    best_ndcg_result = 0.
    best_epoch = 0
    for i in range(start_epoch, start_epoch + additional_epoch):

        scheduler.step()
        net.zero_grad()
        net.train()

        if train_algo == BASELINE:
            epoch_loss = baseline_pairwise_training_loop(
                i, net, loss_func, optimizer,
                train_loader,
                precision=precision, device='cuda:'+str(device), debug=debug
            )
        elif train_algo in [SUM_SESSION, ACC_GRADIENT]:
            epoch_loss = factorized_training_loop(
                i, net, None, optimizer,
                train_loader,
                training_algo=train_algo,
                precision=precision, device='cuda:'+str(device), debug=debug
            )

        losses.append(epoch_loss)
        print('=' * 20 + '\n', get_time(), 'Epoch{}, loss : {}'.format(i, losses[-1]), '\n' + '=' * 20)

        # save to checkpoint every 5 step, and run eval
        if i % 5 == 0 and i != start_epoch:
            save_to_ckpt(ckptfile, i, net, optimizer, scheduler)
            net_inference.load_state_dict(net.state_dict())
            ndcg_result = eval_model(net_inference, device, df_valid, valid_loader, i, writer)
            if best_ndcg_result < ndcg_result[10]:
                best_ndcg_result = ndcg_result[10]
                best_epoch = i

    # save the last ckpt
    save_to_ckpt(ckptfile, start_epoch + additional_epoch, net, optimizer, scheduler)

    # final evaluation
    net_inference.load_state_dict(net.state_dict())
    ndcg_result = eval_model(
        net_inference, device, df_valid, valid_loader, start_epoch + additional_epoch, writer)
    if best_ndcg_result < ndcg_result[10]:
        best_ndcg_result = ndcg_result[10]
        best_epoch = start_epoch + additional_epoch
        
    # save the final model
    torch.save(net.state_dict(), ckptfile)
    print(
        get_time(),
        "finish training " + ", ".join(
            ["NDCG@{}: {:.5f}".format(k, ndcg_result[k]) for k in ndcg_result]
        ),
        '\n\n'
    )
    return best_ndcg_result, best_epoch


def eval_model(inference_model, device, df_valid, valid_loader, epoch, writer=None):
    """
    :param torch.nn.Module inference_model:
    :param str device: cpu or cuda:id
    :param pandas.DataFrame df_valid:
    :param valid_loader:
    :param int epoch:
    :return:
    """
    inference_model.eval()  # Set model to evaluate mode
    batch_size = 1000000

    with torch.no_grad():
        #eval_cross_entropy_loss(inference_model, device, valid_loader, epoch, writer)
        ndcg_result, _ = eval_ndcg_at_k(
            inference_model, device, df_valid, valid_loader, [10, 30], batch_size)
    return ndcg_result


def eval_ndcg_at_k(inference_model, device, df_valid, valid_loader, k_list=[5, 10, 30], batch_size=1000000, phase="Eval"):
    ndcg_metrics = {k: NDCG(k) for k in k_list}
    qids, rels, scores = [], [], []
    inference_model.eval()
    session_ndcgs = defaultdict(list)
    with torch.no_grad():
        for i, (X, Y) in enumerate(valid_loader.generate_batch_per_query()):
            if X is None or X.size()[0] < 2:
                continue
            y_tensor = inference_model.forward(X.to(torch.float32))
            score = y_tensor.cpu().numpy().squeeze()
            rel = Y.cpu().numpy()
            if valid_loader.dataset_type in ['naver'] or \
                (valid_loader.dataset_type == 'naver_click' and valid_loader.data_type == 'test'):
                rel = rel + 1
            result_qid = sorted([(s, r) for s, r in zip(score, rel)], key=lambda x: x[0], reverse=True)
            rel_rank = [s[1] for s in result_qid]
            for k, ndcg in ndcg_metrics.items():
                if ndcg.maxDCG(rel_rank) == 0:
                    continue
                ndcg_k = ndcg.evaluate(rel_rank)
                if not np.isnan(ndcg_k):
                    session_ndcgs[k].append(ndcg_k)
            scores.append(score)
            rels.append(rel)
    ndcg_result = {k: np.mean(session_ndcgs[k]) for k in k_list}
    ndcg_result_print = ", ".join(["NDCG@{}: {:.5f}".format(k, ndcg_result[k]) for k in k_list])
    print(get_time(), "{} Phase evaluate {}".format(phase, ndcg_result_print))
    return ndcg_result, (scores, rels)
    

def get_train_inference_net(train_algo, num_features, start_epoch, double_precision, opt=None, log=True):
    ranknet_structure = [num_features, 64, 16]

    if train_algo == BASELINE:
        net = RankNetPairs(ranknet_structure, double_precision)
        net_inference = RankNet(ranknet_structure)  # inference always use single precision
        ckptfile = get_ckptdir('ranknet', ranknet_structure, opt=opt, log=log)

    elif train_algo in [SUM_SESSION, ACC_GRADIENT]:
        net = RankNet(ranknet_structure, double_precision)
        net_inference = net
        ckptfile = get_ckptdir('ranknet-factorize', ranknet_structure, opt=opt, log=log)

    else:
        raise ValueError("train algo {} not implemented".format(train_algo))

    if start_epoch != 0:
        load_from_ckpt(ckptfile, start_epoch, net, log)

    return net, net_inference, ckptfile



def get_ckptdir(net_name, net_structure, sigma=None, opt=None, log=True):
    net_name = '{}-{}'.format(net_name, '-'.join([str(x) for x in net_structure]))
    if sigma:
        net_name += '-scale-{}'.format(sigma)
    ckptdir = os.path.join('model', 'ckptdir')
    if opt is not None:
        ckptdir = os.path.join(ckptdir, opt)
    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir)    
    ckptfile = os.path.join(ckptdir, net_name)
    if log:
        print("checkpoint dir:", ckptfile)
    return ckptfile


# load model with checkpoint
def get_model(train_loader, ckpt_epoch=50, train_algo=SUM_SESSION, double_precision=False, opt=None, device=0):
    net, net_inference, ckptfile = get_train_inference_net(
        train_algo, train_loader.num_features, ckpt_epoch, double_precision, opt
    )
#     device = "cuda:1"#get_device('RankNet')
#     net.to(device)
#     net_inference.to(device)
    net.cuda(device)
    return net, net


def clear_mislabel(data_loader):
    data_loader.mislabeled_on = False
    data_loader.mislabeled_dict = None


def build_mislabeled_dataset(data_loader, error_query_ratio, error_doc_ratio, error_type):
    clear_mislabel(data_loader)
    assert 0 <= error_query_ratio and error_query_ratio <= 100
    # doc ratio is % based
    assert 0 <= error_doc_ratio and error_doc_ratio <= 100
    assert error_type in ['RAND', 'FN', 'FP', 'CE', 'CE2', 'RAND2', 'SW', 'SWO', \
                          'CE3', 'SW2', 'SW3', 'CE2v3pn', 'CE2v3np', 'SWDIST', 'SWDIST2']
    if error_type == 'SWDIST2':
        error_type = 'SWDIST'
        
    if error_query_ratio == 0 or error_doc_ratio == 0:
        print('Error query ratio:', str(error_query_ratio)+'%',\
                '\tError doc ratio:', str(error_doc_ratio)+'%')
        return 
    else:
        print('Error query ratio:', str(error_query_ratio)+'%',\
                '\tError doc ratio:', str(error_doc_ratio)+'%',\
                '\tError type:', error_type)
    
    data_loader.get_qids()
    data_loader.get_cached_batch_per_query(data_loader.df, data_loader.qids)
    index_list = list(range(data_loader.num_sessions))
    
    clear_seed_all()
    random.shuffle(index_list)
    #if error_type == 'CE2' or error_type == 'SW2':
    if error_type == 'SW2':
        error_query_index = []
        for i in index_list:
            if 3 in data_loader.Y[i] or 4 in data_loader.Y[i]:
                error_query_index.append(i)
        error_query_index = error_query_index[:int(data_loader.num_sessions * error_query_ratio // 100)]
    elif error_type == 'CE2' or error_type == 'SW3' or error_type == 'CE2v3pn' or error_type == 'CE2v3np':
        error_query_index = []
        for i in index_list:
            if 4 in data_loader.Y[i]:
                error_query_index.append(i)
        error_query_index = error_query_index[:int(data_loader.num_sessions * error_query_ratio // 100)]
    else:
        error_query_index = index_list[:int(data_loader.num_sessions * error_query_ratio // 100)]
    if error_type == 'SWDIST':
        distribution = [0, 0, 0, 0, 0]
        for Y in data_loader.Y:
            for i in range(5):
                distribution[i] += (Y == i).nonzero().size()[0]
        distribution = np.array(distribution, np.double)
        print('distribution:', [round(d/distribution.sum(), 4) for d in distribution])
    else:
        distribution = None
    #qids = [full_qids[i] for i in error_query_index]    
    mislabeled_dict = {}
    if error_type == 'RAND2':
        error_query_index = tqdm(error_query_index)
    for i in error_query_index:
        mislabeled_dict[str(i)] = build_error(data_loader.Y[i], error_doc_ratio, error_type, distribution)
    
    data_loader.build_mislabeled(mislabeled_dict, mislabeled_type=error_type)

    
def build_error(Y, error_doc_ratio, error_type, distribution=None):
    if error_type == 'RAND2':
        #relevance가 0이 아닌 pair를 shuffle하여 error_doc_ratio 만큼의 index를 저장
        #TBD
        rel_diff = Y.view(-1, 1) - Y.view(-1, 1).t()
        mislabeled_rel_diff = rel_diff.clone()
        non_neg_index_list = (rel_diff >= 0.).nonzero().data.tolist()
        for self_rel in [[i, i] for i in range(Y.view(-1).size()[0])]:
            non_neg_index_list.remove(self_rel)
        #assert if all the document label is the same
        assert len(non_neg_index_list) > 0
        error_doc_num = max(len(non_neg_index_list) * error_doc_ratio // 100, 1)
        random.shuffle(non_neg_index_list)

        for i, j in non_neg_index_list[:error_doc_num]:
            #+ => -
            #- => +
            assert mislabeled_rel_diff[i, j] == -mislabeled_rel_diff[j, i]
            if rel_diff[i, j] == 0.:
                cand = [-1., 1.]
                mislabeled_rel_diff[i, j] = random.choice(cand)
                mislabeled_rel_diff[j, i] = mislabeled_rel_diff[i, j] * -1.
            elif rel_diff[i, j] > 0.:
                cand = [-1., 0.]
                mislabeled_rel_diff[i, j] = mislabeled_rel_diff[i, j] * random.choice(cand)
                mislabeled_rel_diff[j, i] = mislabeled_rel_diff[i, j] * -1.
            else:
                raise NotImplementedError
        return mislabeled_rel_diff

    
    mislabeled_Y = Y.clone()
    if error_type == 'RAND':
        #original label이 아닌 무언가로 random하게 변화
        original_label = [0, 1, 2, 3, 4]
    elif error_type == 'FN':
        #2,3,4 => 0,1
        original_label = [2, 3, 4]
    elif error_type == 'FP':
        #0,1 => 2,3,4
        original_label = [0, 1]
    elif error_type == 'CE':
        #0 => 4 / 4 => 0
        original_label = [0, 4]
    elif error_type == 'CE2':
        #0 => 3, 4 / 3, 4 => 0
        original_label = [3, 4]
        neg_label = [0]
        neg_index_list = [idx for idx in range(len(Y)) if Y[idx] in neg_label]
        random.shuffle(neg_index_list)
    elif error_type == 'CE3':
        #0 => 2, 3, 4 / 2, 3, 4 => 0
        original_label = [2, 3, 4]
        neg_label = [0]
        neg_index_list = [idx for idx in range(len(Y)) if Y[idx] in neg_label]
        random.shuffle(neg_index_list)
    elif error_type == 'SW' or error_type == 'SWO':
        original_label = [2, 3, 4]
        neg_label = [0, 1]
        neg_index_list = [idx for idx in range(len(Y)) if Y[idx] in neg_label]
        random.shuffle(neg_index_list)
    elif error_type == 'SW2' or error_type == 'SW3':
        original_label = [3, 4]
        neg_label = [0, 1]
        neg_index_list = [idx for idx in range(len(Y)) if Y[idx] in neg_label]
        random.shuffle(neg_index_list)
    elif error_type == 'CE2v3pn':
        #3, 4 => 0
        original_label = [3, 4]
    elif error_type == 'CE2v3np':
        #0 => 3, 4
        original_label = [3, 4]
        neg_label = [0]
        neg_index_list = [idx for idx in range(len(Y)) if Y[idx] in neg_label]
        random.shuffle(neg_index_list)
    elif error_type == 'SWDIST':
        #2, 3, 4 => 0, 1 / 0, 1 => 2, 3, 4 | train distribution
        assert distribution is not None
        original_label = [2, 3, 4]
        neg_label = [0, 1]
        neg_index_list = [idx for idx in range(len(Y)) if Y[idx] in neg_label]
        random.shuffle(neg_index_list)
    else:
        raise NotImplementedError
        
    index_list = [idx for idx in range(len(Y)) if Y[idx] in original_label]
    #max(..., 0)이어야 하나..?
    #query 쪽이 0%면 어차피 여기까지 안오긴 함
    error_doc_num = max(len(index_list) * error_doc_ratio // 100, 1)
    random.shuffle(index_list)
    
    if error_type == 'SW' or error_type == 'SWO' or error_type == 'SW2' or error_type == 'SW3':
        if error_type == 'SWO':
            #4, 3, 2 순으로 Switch
            ordered_index_list = []
            for l in sorted(original_label, reverse=True):
                ordered_index_list.extend([idx for idx in index_list if Y[idx] == l])
            assert len(ordered_index_list) == len(index_list)
            index_list = ordered_index_list
        for i, (p_idx, n_idx) in enumerate(zip(index_list[:error_doc_num], 
                                               neg_index_list[:error_doc_num])):
            assert Y[p_idx] in original_label
            assert Y[n_idx] in neg_label
            # 2, 3, 4 => 0, 1 / 0, 1 => 2, 3, 4 (Switch)
            mislabeled_Y[p_idx] = Y[n_idx].item()
            mislabeled_Y[n_idx] = Y[p_idx].item()
            
        return mislabeled_Y
    
    
    if error_type == 'SWDIST':
        error_neg_doc_num = max(len(neg_index_list) * error_doc_ratio // 100, 1)
        pos_distribution = np.array([distribution[2], distribution[3], distribution[4]])
        pos_distribution = pos_distribution / pos_distribution.sum()
        neg_distribution = np.array([distribution[0], distribution[1]])
        neg_distribution = neg_distribution / neg_distribution.sum()
        
        for idx in index_list[:error_doc_num]:
            assert Y[idx] in original_label
            mislabeled_Y[idx] = int(np.random.choice([0, 1], 1, p=neg_distribution)[0])
            
        for idx in neg_index_list[:error_neg_doc_num]:
            assert Y[idx] in neg_label
            mislabeled_Y[idx] = int(np.random.choice([2, 3, 4], 1, p=pos_distribution)[0])
            
        return mislabeled_Y
    
    
    for idx in index_list[:error_doc_num]:
        assert Y[idx] in original_label
        if error_type == 'CE2v3np':
            break
        if error_type == 'RAND':
            #original label이 아닌 무언가로 random하게 변화
            cand = [0, 1, 2, 3, 4]
            cand.remove(Y[idx])            
        elif error_type == 'FN':
            #2,3,4 => 0,1
            cand = [0, 1]
        elif error_type == 'FP':
            #0,1 => 2,3,4
            cand = [2, 3, 4]
        elif error_type == 'CE':
            #0 => 4 / 4 => 0
            if Y[idx] == 0:
                cand = [4]
            elif Y[idx] == 4:
                cand = [0]            
        elif error_type == 'CE2':
            #0 => 3, 4 / 3, 4 => 0
            cand = [0]
        elif error_type == 'CE3' or error_type == 'CE2v3pn':
            #0 => 2, 3, 4 / 2, 3, 4 => 0
            cand = [0]
        mislabeled_Y[idx] = random.choice(cand)
        
    if error_type == 'CE2' or error_type == 'CE3' or error_type == 'CE2v3np':
        for i, idx in enumerate(neg_index_list[:error_doc_num]):
            assert Y[idx] in neg_label
            mislabeled_Y[idx] = Y[index_list[i]].item()

            
    return mislabeled_Y


def get_lambda_grad(y_pred, Y, pairs, precision=torch.float32, sigma=1.0, ndcg_gain_in_train="exp2"):
    # compute the rank order of each document
    Y_list = Y.data.tolist()
    ideal_dcg = NDCG(2**9, ndcg_gain_in_train)
    N = 1.0 / ideal_dcg.maxDCG(Y_list)
    Y = Y.to(precision)    
    
    rank_df = pd.DataFrame({"Y": Y_list, "doc": np.arange(Y.shape[0])})
    rank_df = rank_df.sort_values("Y").reset_index(drop=True)
    rank_order = rank_df.sort_values("doc").index.values + 1

    
    device = y_pred.get_device()
    with torch.no_grad():
        pairs_score_diff = 1.0 + torch.exp(sigma * (y_pred - y_pred.t()))
        rel_diff = Y - Y.t()
        neg_pairs = (rel_diff < 0).type(precision)
        Sij = pairs - neg_pairs
        gain_diff = torch.pow(2.0, Y) - torch.pow(2.0, Y.t())

        rank_order_tensor = torch.tensor(rank_order, dtype=precision, device=device).view(-1, 1)
        decay_diff = 1.0 / torch.log2(rank_order_tensor + 1.0) - 1.0 / torch.log2(rank_order_tensor.t() + 1.0)

        delta_ndcg = torch.abs(N * gain_diff * decay_diff)
        lambda_update = sigma * (0.5 * (1 - Sij) - 1 / pairs_score_diff) * delta_ndcg
        lambda_update = torch.sum(lambda_update, 1, keepdim=True)

        assert lambda_update.shape == y_pred.shape
        check_grad = torch.sum(lambda_update, (0, 1)).item()
        if check_grad == float('inf') or np.isnan(check_grad):
            import ipdb; ipdb.set_trace()
    return lambda_update


def factorized_training_loop(
    epoch, net, loss_func, optimizer,
    train_loader, batch_size=200, sigma=1.0,
    training_algo=SUM_SESSION,
    precision=torch.float32, device="cpu",
    debug=False,
    LambdaRank=False
):
    print(training_algo)
    minibatch_loss = []
    count, loss, total_pairs = 0, 0, 0
    grad_batch, y_pred_batch = [], []
    
    tmp_idx_order = []
    for X, Y in train_loader.generate_batch_per_query(shuffle=True):
        ###############################
        tmp_idx_order.append(train_loader.current_idx)
        #continue
        ###############################
        if X is None or X.shape[0] == 0:
            continue
        Y = Y.view(-1, 1)
        rel_diff = Y - Y.t()
        
        #Handling pairwise relevance mislabel
        #TBD binary label will not be applied
        if train_loader.mislabeled_type == 'RAND2' and train_loader.mislabeled_on \
                            and (train_loader.mislabeled_dict is not None) \
                and (str(train_loader.current_idx) in train_loader.mislabeled_dict.keys()):
            #print('RAND2 is working')
            m_rel_diff = train_loader.mislabeled_dict[str(train_loader.current_idx)]
            assert (rel_diff - m_rel_diff).nonzero().sum() > 0
            rel_diff = m_rel_diff
            
        #Handling document drop
        if train_loader.current_idx in train_loader.drop_documents.keys():
            for drop_doc_idx in train_loader.drop_documents[train_loader.current_idx]:
                rel_diff[drop_doc_idx, :] = 0
                rel_diff[:, drop_doc_idx] = 0
        #Handling document drop
        if train_loader.current_idx in train_loader.drop_pairs.keys():
            for (drop_doc1, drop_doc2) in train_loader.drop_pairs[train_loader.current_idx]:
                rel_diff[drop_doc1, drop_doc2] = 0
                rel_diff[drop_doc2, drop_doc1] = 0
                
        pairs = (rel_diff > 0).to(precision)
        num_pairs = torch.sum(pairs, (0, 1))
        # skip negative sessions, no relevant info:
        if num_pairs == 0:
            continue

        X_tensor = X.to(precision)
           
        y_pred = net(X_tensor)

        if training_algo == SUM_SESSION:
            #2020.05.07
            if LambdaRank:
                y_pred_batch.append(y_pred)
                lambda_update = get_lambda_grad(y_pred, Y, pairs, precision=precision)
                grad_batch.append(lambda_update)
            #LambdaRank: DO SOMETHING
            else:
                C = criterion(y_pred, pairs)
                loss += torch.sum(C)
        else:
            raise ValueError("training algo {} not implemented".format(training_algo))

        total_pairs += num_pairs
        count += 1

        if count % batch_size == 0:
            loss /= total_pairs
            minibatch_loss.append(loss.item())
            if debug:
                print("Epoch {}, number of pairs {}, loss {}".format(epoch, total_pairs, loss.item()))
            if training_algo == SUM_SESSION:
                if USE_AMP:
                    pass
#                     with amp.scale_loss(loss, optimizer) as scaled_loss:
#                         scaled_loss.backward()
                else:
                    if LambdaRank:
                        for grad, y_pred in zip(grad_batch, y_pred_batch):
                            y_pred.backward(grad / batch_size)
                        
                    else:
                        loss.backward()
            elif training_algo == ACC_GRADIENT:
                for grad, y_pred in zip(grad_batch, y_pred_batch):
                    y_pred.backward(grad / batch_size)

            if count % (4 * batch_size) and debug:
                net.dump_param()

            optimizer.step()
            net.zero_grad()
            loss, total_pairs = 0, 0  # loss used for sum_session
            grad_batch, y_pred_batch = [], []  # grad_batch, y_pred_batch used for gradient_acc
            #torch.cuda.empty_cache()
    #print(tmp_idx_order[:10])
    if total_pairs:
        print('+' * 10, "End of batch, remaining pairs {}".format(total_pairs.item()))
        loss /= total_pairs
        minibatch_loss.append(loss.item())
        if training_algo == SUM_SESSION:
            if USE_AMP:
                pass
#                 with amp.scale_loss(loss, optimizer) as scaled_loss:
#                     scaled_loss.backward()
            else:
                if LambdaRank:
                    for grad, y_pred in zip(grad_batch, y_pred_batch):
                        y_pred.backward(grad / batch_size)
                else:
                    loss.backward()
        else:
            for grad, y_pred in zip(grad_batch, y_pred_batch):
                y_pred.backward(grad / total_pairs)

        if debug:
            net.dump_param()
        optimizer.step()

        return np.mean(minibatch_loss)
    
    
    
#================================================================
#INFLUENCE FUNCTIONS
#================================================================

# 같은 query에 대한 모든 document pair loss를 반환
def get_prediction(X, Y, net, data_loader, precision=torch.float32):
    if X is None or X.size()[0] == 0:
        return None, None
    #Handling pairwise relevance mislabel
    #TBD binary label will not be applied
    if data_loader.mislabeled_type == 'RAND2' and data_loader.mislabeled_on \
                        and (data_loader.mislabeled_dict is not None) \
            and (str(data_loader.current_idx) in data_loader.mislabeled_dict.keys()):
        rel_diff = data_loader.mislabeled_dict[str(data_loader.current_idx)]
    else:
        Y = Y.view(-1, 1)
        rel_diff = Y - Y.t()
    #del Y
    if data_loader.current_idx in data_loader.drop_documents.keys():
        for drop_doc_idx in data_loader.drop_documents[data_loader.current_idx]:
            rel_diff[drop_doc_idx, :] = 0
            rel_diff[:, drop_doc_idx] = 0
    #Handling document drop
    if data_loader.current_idx in data_loader.drop_pairs.keys():
        for (drop_doc1, drop_doc2) in data_loader.drop_pairs[data_loader.current_idx]:
            rel_diff[drop_doc1, drop_doc2] = 0
            rel_diff[drop_doc2, drop_doc1] = 0

    pos_pairs = (rel_diff > 0).to(precision)
    num_pos_pairs = torch.sum(pos_pairs, (0, 1))
    if num_pos_pairs == 0:
        return None, None#, None
    if num_pos_pairs == 0:
        return None, None#, None
    
    #neg_pairs = (rel_diff < 0).to(precision)
    #num_pairs = 2 * num_pos_pairs  # num pos pairs and neg pairs are always the same

    #X_tensor = X.to(torch.float32)#torch.tensor(X, dtype=precision, device=device)
    y_pred = net(X.to(precision))
    #del X
    #torch.cuda.empty_cache()

    return y_pred, pos_pairs#, neg_pairs
    
def criterion(y_pred, pairs, sigma=1.0, precision=torch.float32):
    damping = 1e-8
    C = torch.log(1 + torch.exp(-sigma * torch.sigmoid(y_pred - y_pred.t())) + damping).to(precision)
    loss = pairs * C
            
    return loss


model_name = 'RankNet'

def get_loss(model, data_loader, criterion, indices, ij_index=None, bar=False, precision=torch.float32):
    losses = []
    if bar:
        indices = tqdm(indices)
    cnt = 0
    for idx in indices:
        cnt += 1
        X, Y = data_loader.indexing_batch_per_query(idx)
        y_pred, pairs = get_prediction(X, Y, model, data_loader, precision=precision)
        if y_pred is None:
            losses.append(torch.tensor([]).to(list(model.parameters())[0].get_device()))
            continue
        loss = criterion(y_pred, pairs, precision=precision)
        del y_pred
        if ij_index is not None:
            _pairs = torch.zeros(pairs.size())
            _pairs[ij_index] = pairs[ij_index]
            pairs = _pairs
        # 여기에 weighting 가능
        # loss = loss * weight
        losses.append(loss[pairs.bool()])
        #print(losses[-1].dtype, len(losses[-1]), len(losses), sum([len(l) for l in losses]))
        torch.cuda.empty_cache()
    return losses


def get_loss_in_the_same_query(model, data_loader, criterion, indices, bar=False, precision=torch.float32):
    losses = []
    if bar:
        indices = tqdm(indices)

    for idx in indices:
        X, Y = data_loader.indexing_batch_per_query(idx)
        y_pred, pairs = get_prediction(X, Y, model, data_loader, precision=precision)
        if y_pred is None:
            continue
        loss = criterion(y_pred, pairs, precision=precision)
        for ij_index in range(len(Y)):
            _pairs = torch.zeros(pairs.size())
            _pairs[ij_index] = pairs[ij_index]
            losses.append(loss[_pairs.bool()])
        
    return losses


def get_query_loss(model, data_loader, criterion, indices, bar=False, precision=torch.float32):
    losses = []
    if bar:
        indices = tqdm(indices)

    for idx in indices:
        X, Y = data_loader.indexing_batch_per_query(idx)
        y_pred, pairs = get_prediction(X, Y, model, data_loader, precision=precision)
        if y_pred is None:
            continue
        loss = criterion(y_pred, pairs, precision=precision)
        losses.append(loss)
        #losses.append(loss.sum())
        
    return losses

def get_doc_loss(model, data_loader, criterion, indices, bar=False, precision=torch.float32):
    losses = []
    if bar:
        indices = tqdm(indices)

    for idx in indices:
        X, Y = data_loader.indexing_batch_per_query(idx)
        y_pred, pairs = get_prediction(X, Y, model, data_loader, precision=precision)
        if y_pred is None:
            continue
        loss = criterion(y_pred, pairs, precision=precision)
        all_loss = loss + loss.t()
        losses.append(all_loss.sum(dim=1))        
        #losses.append(all_loss.mean(dim=1))        
    return losses
    
    
def get_pair_loss(model, data_loader, criterion, indices, bar=False, precision=torch.float32):
    losses = []
    if bar:
        indices = tqdm(indices)

    for idx in indices:
        X, Y = data_loader.indexing_batch_per_query(idx)
        y_pred, pairs = get_prediction(X, Y, model, data_loader, precision=precision)
        if y_pred is None:
            continue
        loss = criterion(y_pred, pairs, precision=precision)
        losses.append(loss)
        
    return losses
    

def get_grad_loss_no_reg_val(trained_model, data_loader, criterion, indices, ij_index=None, 
                             query_loss=True, individual_weight=False, mean=True, bar=False, losses=None):
    params = trained_model.parameters()
#     print("get_grad_loss_no_reg_val params", sum(p.numel() for p in params if p.requires_grad))
#     print("get_grad_loss_no_reg_val model.parameters()", sum(p.numel() for p in trained_model.parameters() if p.requires_grad))
    grad_loss_no_reg_val = None    
    
    if losses is None:
        assert indices is not None
        losses = get_loss(trained_model, data_loader, criterion, indices, ij_index, bar)    
    empty_loss = 0
    for loss in losses:
        if len(loss) == 0 or (loss == 0.).int().sum() == len(loss):
            empty_loss += 1
            continue
        if not individual_weight: #calcutate same query losses all at once
            grad = autograd.grad(loss.sum(), trained_model.parameters(), retain_graph=True)
            grad = list(grad)
        else:
            grad = None
            for l in tqdm(loss.view(-1)):
                _grad = autograd.grad(l, trained_model.parameters(), retain_graph=True)
                raise NotImplementedError
                # individual ij grad에 weighting
                #_grad = [a * weight for a in _grad]
                with torch.no_grad():
                    if grad is None:
                        grad = _grad
                    else:
                        grad = [a + b for (a, b) in zip(grad, _grad)]
        # 각 query 별로 grad 평균
        if query_loss:
            grad = [a/loss.view(-1).size()[0] for a in grad]
            
        with torch.no_grad():
            if grad_loss_no_reg_val is None: # 'initialized' at first call
                grad_loss_no_reg_val = grad
            else:
                grad_loss_no_reg_val = [a + b for (a, b) in zip(grad_loss_no_reg_val, grad)]

    if mean:
        if query_loss: # query 별 grad 평균
            grad_loss_no_reg_val = [a/(len(losses)-empty_loss) for a in grad_loss_no_reg_val]    
        else:
            grad_loss_no_reg_val = [a/sum([len(loss) for loss in losses]) for a in grad_loss_no_reg_val]

    return grad_loss_no_reg_val


def get_lambda_param_grad(model, grad_batch, y_pred_batch):
    model.zero_grad()
    for grad, y_pred in zip(grad_batch, y_pred_batch):
        y_pred.backward(grad)
    param_grad_list = []
    for param in model.parameters():
        param_grad_list.append(param.grad.detach().clone())
    model.zero_grad()
    
    return param_grad_list


# def get_lambda_grad(model, grad_batch, y_pred_batch):
#     get_lambda_param_grad(model, grad_batch, y_pred_batch)
#     assert 1 == 2


#====================================================================================================================
# NEW s_test code

from torch.autograd import grad

def get_s_test(z_test_grad, z_losses, params, damp=0.01, scale=25.0, recursion_depth=20000, threshold=1e-8):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, strochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.

    Arguments:
        z_test: torch tensor, test data points, such as test images
        t_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        z_loader: torch Dataloader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.

    Returns:
        h_estimate: list of torch tensors, s_test"""
    v = z_test_grad
    h_estimate = v.copy()
    tmp_h_estimate = h_estimate.copy()
    
    num_of_z_losses = sum([z_loss.view(-1).size()[0] for z_loss in z_losses])
    num_of_params = sum([param.view(-1).size()[0] for param in params])
    
    #############################
    #just use the DataLoader once
    #############################
    random.seed(7777)
    z_losses_idx = list(range(len(z_losses)))
    random.shuffle(z_losses_idx)
    for i in tqdm(range(recursion_depth)):
        random.shuffle(z_losses_idx)
        #z_loss_idx = list(range(len(z_losses[0])))
        z_pick = 0
        for idx in z_losses_idx:
            if z_losses[idx].size()[0] > 0:
                z_pick = idx
                break
        z_loss_idx = list(range(len(z_losses[z_pick])))
        random.shuffle(z_loss_idx)
        for j in z_loss_idx:
            hv = get_hvp(z_losses[z_pick][j], params, h_estimate)
            #hv = get_hvp(z_losses[j], params, h_estimate)
            # Recursively caclulate h_estimate
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / (scale * num_of_z_losses)
                for _v, _h_e, _hv in zip(v, h_estimate, hv)]
            break
        if i % 1000 == 0:
            print([v[0] for v in h_estimate])
            overlap = sum([((tmp_h_e.view(-1) - _h_e.view(-1)).abs() < threshold).int().sum().item() 
                           for tmp_h_e, _h_e in zip(tmp_h_estimate, h_estimate)])
            if overlap == num_of_params:
                break
            tmp_h_estimate = h_estimate.copy()
    #print(h_estimate)
    return h_estimate

def get_hvp(y, w, v):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.

    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian

    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.

    Raises:
        ValueError: `y` and `w` have a different length."""
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True, allow_unused=True)
    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, retain_graph=True, create_graph=False, allow_unused=True)

    return return_grads

#====================================================================================================================


VERBOSE = False

# input: model, trainset, testset, loss function, test point
# output: test point의 loss function에 대한 influence
def get_influence_on_test_loss(trained_model, train_set, test_set, criterion, test_indices,
                               force_refresh=True,
                               approx_filename='', 
                               losses=None, 
                               query_level=False,
                               pair_level=False,
                               device=0,
                               q_mean=False
                               ):
    torch.cuda.set_device(device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


    #GET s_test
    start_time = time.time()
    #q_mean = False
    if q_mean:
#         z_test_losses = get_loss_in_the_same_query(trained_model, test_set, criterion, test_indices, bar=True)
        z_test_losses = get_loss(trained_model, test_set, criterion, test_indices, bar=True)
        
        z_test_grad = get_grad_loss_no_reg_val(trained_model, test_set, criterion, mean=False, query_loss=True,
                                                indices=None, losses=z_test_losses)
    else:
        z_test_losses = get_loss(trained_model, test_set, criterion, test_indices, bar=True)
        
        z_test_loss = params_to_tensor(z_test_losses) # make list of tensors to a tensor
        
        z_test_grad = get_grad_loss_no_reg_val(trained_model, test_set, criterion, mean=False, query_loss=False,
                                                indices=None, losses=z_test_loss.view(1, 1, -1))
        
    z_losses = get_loss(trained_model, train_set, criterion, list(range(0, train_set.num_sessions)), bar=True)
    params = list(trained_model.parameters())

    inverse_hvp = get_s_test(z_test_grad, z_losses, params, recursion_depth=20000)
    inverse_hvp = params_to_tensor(inverse_hvp)
    
    duration = time.time() - start_time
    print('Inverse HVP took %s sec' % duration)

    start_time = time.time()
    
    num_of_z_losses = sum([z_loss.view(-1).size()[0] for z_loss in z_losses])
    num_to_remove = train_set.num_sessions
    if pair_level:
        predicted_loss_diffs = []
        inverse_hvp = torch.tensor(inverse_hvp).cuda().float()
        for i in tqdm(range(num_to_remove)):
            _, Y = train_set.indexing_batch_per_query(idx=i)
            qid = train_set.df.qid.unique()[i]
            losses = get_pair_loss(trained_model, train_set, criterion, [i])
            if len(losses) == 0:
                predicted_loss_diffs.append([])
                continue
            losses = losses[0]
            _predicted_loss_diffs = []
            for j in range(len(Y)):
                for k in range(len(Y)):
                    train_grad_loss_val = None
                    if losses[j][k] != 0.:
                        train_grad_loss_val = get_grad_loss_no_reg_val(trained_model, train_set, criterion, [i], ij_index=j, 
                                                                       query_loss=False, mean=False, losses=losses[j][k].view(1, 1, 1))
                    if train_grad_loss_val is None:
    #                     if i in train_set.drop_documents.keys() and j in train_set.drop_documents[i]:
    #                         print(i,j, 'skipped')
                        _predicted_loss_diffs.append(float(np.inf))
                        continue
                    train_grad_loss_val = params_to_tensor(train_grad_loss_val)
                    _predicted_loss_diffs.append(torch.dot(inverse_hvp, train_grad_loss_val).item() / num_of_z_losses)#train_set.num_pairs
                
            predicted_loss_diffs.append(np.array(_predicted_loss_diffs).reshape(len(Y), len(Y)))  
    elif query_level:
        predicted_loss_diffs = []
        for i in tqdm(range(num_to_remove)):
            _, Y = train_set.indexing_batch_per_query(idx=i)
            qid = train_set.df.qid.unique()[i]
            losses = get_query_loss(trained_model, train_set, criterion, [i])
            if len(losses) == 0:
                predicted_loss_diffs.append(float(np.inf))
                continue
            losses = losses[0]
            train_grad_loss_val = get_grad_loss_no_reg_val(trained_model, train_set, criterion, [i], 
                                                           query_loss=True, mean=False, losses=losses.view(1, 1, -1))
            if train_grad_loss_val is None:
                predicted_loss_diffs.append(float(np.inf))
                continue
            train_grad_loss_val = params_to_tensor(train_grad_loss_val)
            predicted_loss_diffs.append(torch.dot(inverse_hvp, train_grad_loss_val).item() / num_of_z_losses)
    elif False: # Random-group
        predicted_loss_diffs = []
        inverse_hvp = torch.tensor(inverse_hvp).cuda().float()
        group_size = 100
        import copy
        
        random_group = {}

        _idx_list = []
        _loss_list = []
        for i in tqdm(range(group_size)):
            _, Y = train_set.indexing_batch_per_query(idx=i)
            losses = get_pair_loss(trained_model, train_set, criterion, [i])
            if len(losses) == 0:
                continue
            losses = losses[0]
            for j in range(len(Y)):
                for k in range(len(Y)):
                    if losses[j][k] != 0.:
                        _idx_list.append((i, j, k))
                        _loss_list.append(losses[j][k])
        index_for_shuffle = list(range(len(_idx_list)))

        clear_seed_all()
        random.shuffle(index_for_shuffle)

        idx_list = []
        infl_list = []
        gs = int(len(_idx_list)/100)
        for i in tqdm(range(gs)):
            tmp_idx_list = []
            tmp_loss_list = None
            for idx in index_for_shuffle[i*100:(i+1)*100]:
                tmp_idx_list.append(_idx_list[idx])
                if tmp_loss_list is None:
                    tmp_loss_list = _loss_list[idx].view(1)
                else:
                    tmp_loss_list = torch.cat([tmp_loss_list, _loss_list[idx].view(1)], dim=0)
            idx_list.append(tmp_idx_list)
            train_grad_loss_val = get_grad_loss_no_reg_val(trained_model, train_set, criterion, [i], 
                                                           query_loss=False, mean=False, losses=tmp_loss_list.view(1, 1, -1))
            if train_grad_loss_val is None:
                infl_list.append(float(np.inf))
                continue
            train_grad_loss_val = params_to_tensor(train_grad_loss_val)
            infl_list.append(torch.dot(inverse_hvp, train_grad_loss_val).item() / num_of_z_losses)
            
        infl_list_with_idx = [(i, infl) for i, infl in enumerate(infl_list)]

        sorted_infl_list = sorted(infl_list_with_idx, key=lambda x: x[1])[:100]

        meta_drop_documents = []
        meta_drop_infl = []
        for i, tmp_infl in sorted_infl_list:
            drop_documents = {}
            for q_idx, doc_idx, doc_idx2 in idx_list[i]:
                if q_idx in drop_documents.keys():
                    drop_documents[q_idx].append((doc_idx, doc_idx2))
                else:
                    drop_documents[q_idx] = [(doc_idx, doc_idx2)]
            meta_drop_documents.append(drop_documents)
            meta_drop_infl.append(tmp_infl)

#         with open('RGS_'+'20_70_SWDIST2'+'.pkl', 'wb') as fp:
#             pickle.dump((meta_drop_documents, meta_drop_infl), fp, pickle.HIGHEST_PROTOCOL)
            
#         assert 1 == 2
    else:
        predicted_loss_diffs = []
        inverse_hvp = torch.tensor(inverse_hvp).cuda().float()
        for i in tqdm(range(num_to_remove)):
            _, Y = train_set.indexing_batch_per_query(idx=i)
            qid = train_set.df.qid.unique()[i]
            losses = get_doc_loss(trained_model, train_set, criterion, [i])
            if len(losses) == 0:
                predicted_loss_diffs.append([])
                continue
            losses = losses[0]
            _predicted_loss_diffs = []
            for j in range(len(Y)):
                train_grad_loss_val = get_grad_loss_no_reg_val(trained_model, train_set, criterion, [i], ij_index=j, 
                                                               query_loss=False, mean=False, losses=losses[j].view(1, 1, 1))
                if train_grad_loss_val is None:
#                     if i in train_set.drop_documents.keys() and j in train_set.drop_documents[i]:
#                         print(i,j, 'skipped')
                    _predicted_loss_diffs.append(float(np.inf))
                    continue
                train_grad_loss_val = params_to_tensor(train_grad_loss_val)
                _predicted_loss_diffs.append(torch.dot(inverse_hvp, train_grad_loss_val).item() / (num_of_z_losses))#*len(Y)))#train_set.num_pairs
                
            predicted_loss_diffs.append(_predicted_loss_diffs)

        
            

    #print("train_grad_loss_list", train_grad_loss_list)
    duration = time.time() - start_time
    print('Multiplying by %s train examples took %s sec' % (num_to_remove, duration))

    return predicted_loss_diffs



def params_to_tensor(tensor_list):
    return torch.cat([t.view(-1) for t in tensor_list])


def clear_seed_all(seed=7777):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)