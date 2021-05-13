"""
Microsoft Learning to Rank Dataset:
https://www.microsoft.com/en-us/research/project/mslr/
"""
import datetime
import os

import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle
from tqdm import tqdm
import torch
import random

def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class DataLoader:

    def __init__(self, path):
        """
        :param path: str
        """
        self.path = path
        self.pickle_path = path[:-3] + 'pkl'
        self.df = None
        self.num_pairs = None
        self.num_sessions = None
        data_type = path.split('/')[-1][:-4]
        if data_type == 'vali':
            data_type = 'valid'
        dataset_type = path.split('/')[-3]
        self.new_vars(data_type=data_type, dataset_type=dataset_type)
    
    def new_vars(self, mislabeled_on=False, mislabeled_dict=None, sampled_qids=None, 
                                X=None, small_train=False, device=0, data_type=None, binary_label=False, influences_on=False, influences_path=None, dataset_type='mslr-web30k'):
        self.dataset_type = dataset_type
        self.mislabeled_on = mislabeled_on
        self.mislabeled_dict = mislabeled_dict
        self.mislabeled_type = None
        self.sampled_qids = sampled_qids
        if self.dataset_type in ['mslr-web30k', 'mslr-web10k']:
            self.X = X
        self.small_train = small_train
        self.device = device
        self.data_type = data_type
        self.binary_label = binary_label
        self.qids = None
        self.get_qids()
        self.get_cached_batch_per_query(self.df, self.qids)
        ###########################################
        #NOT USED ANYMORE
        self.influences_on = influences_on
        if self.influences_on:
            assert type(influences_path) == str
            with open(influences_path, 'rb') as fp:
                self.influences = pickle.load(fp)
        else:
            self.influences = None
        ###########################################
        self.current_idx = 0
        if False:
            self.num_features = 68
        elif self.dataset_type in ['mslr-web30k', 'mslr-web10k', 'mq2008-semi']:
            self.num_features = len(self.df.columns) - 2
        elif self.dataset_type in ['naver', 'naver_click']:
            self.num_features = len(self.df.columns) - 4
        self.drop_queries = []
        self.drop_documents = {}
        self.drop_pairs = {}

        
        
    def get_num_pairs(self):
        if self.num_pairs is not None:
            return self.num_pairs
        self.num_pairs = 0
        for _, Y in self.generate_batch_per_query(self.df):
            Y = Y.view(-1, 1)
            pairs = Y - Y.t()
            pos_pairs = torch.sum(pairs > 0, (0, 1))
            neg_pairs = torch.sum(pairs < 0, (0, 1))
            assert pos_pairs == neg_pairs
            self.num_pairs += pos_pairs.item() + neg_pairs.item()
        return self.num_pairs

    def get_num_sessions(self):
        return self.num_sessions

    def _load_mslr(self):
        print(get_time(), "load file from {}".format(self.path))
        df = pd.read_csv(self.path, sep=" ", header=None)
        df.drop(columns=df.columns[-1], inplace=True)
        self.num_features = len(df.columns) - 2
        self.num_paris = None
        print(get_time(), "finish loading from {}".format(self.path))
        print("dataframe shape: {}, features: {}".format(df.shape, self.num_features))
        return df

    def _parse_feature_and_label(self, df):
        """
        :param df: pandas.DataFrame
        :return: pandas.DataFrame
        """
        print(get_time(), "parse dataframe ...", df.shape)
        for col in range(1, len(df.columns)):
            if ':' in str(df.iloc[:, col][0]):
                df.iloc[:, col] = df.iloc[:, col].apply(lambda x: x.split(":")[1])
        df.columns = ['rel', 'qid'] + [str(f) for f in range(1, len(df.columns) - 1)]

        for col in [str(f) for f in range(1, len(df.columns) - 1)]:
            df[col] = df[col].astype(np.float32)

        print(get_time(), "finish parsing dataframe")
        self.df = df
        self.num_sessions = len(df.qid.unique())
        return df

    def generate_query_pairs(self, df, qid):
        """
        :param df: pandas.DataFrame, contains column qid, rel, fid from 1 to self.num_features
        :param qid: query id
        :returns: numpy.ndarray of x_i, y_i, x_j, y_j
        """
        df_qid = df[df.qid == qid]
        rels = df_qid.rel.unique()
        x_i, x_j, y_i, y_j = [], [], [], []
        for r in rels:
            df1 = df_qid[df_qid.rel == r]
            df2 = df_qid[df_qid.rel != r]
            df_merged = pd.merge(df1, df2, on='qid')
            df_merged.reindex(np.random.permutation(df_merged.index))
            y_i.append(df_merged.rel_x.values.reshape(-1, 1))
            y_j.append(df_merged.rel_y.values.reshape(-1, 1))
            x_i.append(df_merged[['{}_x'.format(i) for i in range(1, self.num_features + 1)]].values)
            x_j.append(df_merged[['{}_y'.format(i) for i in range(1, self.num_features + 1)]].values)
        return np.vstack(x_i), np.vstack(y_i), np.vstack(x_j), np.vstack(y_j)

    def generate_query_pair_batch(self, df=None, batchsize=2000, shuffle=False):
        """
        :param df: pandas.DataFrame, contains column qid
        :returns: numpy.ndarray of x_i, y_i, x_j, y_j
        """
        if df is None:
            df = self.df
        x_i_buf, y_i_buf, x_j_buf, y_j_buf = None, None, None, None
        self.get_qids()
        if shuffle:
            np.random.shuffle(self.qids)
        for qid in self.qids:
            x_i, y_i, x_j, y_j = self.generate_query_pairs(df, qid)
            if x_i_buf is None:
                x_i_buf, y_i_buf, x_j_buf, y_j_buf = x_i, y_i, x_j, y_j
            else:
                x_i_buf = np.vstack((x_i_buf, x_i))
                y_i_buf = np.vstack((y_i_buf, y_i))
                x_j_buf = np.vstack((x_j_buf, x_j))
                y_j_buf = np.vstack((y_j_buf, y_j))
            idx = 0
            while (idx + 1) * batchsize <= x_i_buf.shape[0]:
                start = idx * batchsize
                end = (idx + 1) * batchsize
                yield x_i_buf[start: end, :], y_i_buf[start: end, :], x_j_buf[start: end, :], y_j_buf[start: end, :]
                idx += 1

            x_i_buf = x_i_buf[idx * batchsize:, :]
            y_i_buf = y_i_buf[idx * batchsize:, :]
            x_j_buf = x_j_buf[idx * batchsize:, :]
            y_j_buf = y_j_buf[idx * batchsize:, :]

        yield x_i_buf, y_i_buf, x_j_buf, y_j_buf

    def generate_query_batch(self, df, batchsize=100000):
        assert 1 == 2
        """
        :param df: pandas.DataFrame, contains column qid
        :returns: numpy.ndarray qid, rel, x_i
        """
        idx = 0
        while idx * batchsize < df.shape[0]:
            r = df.iloc[idx * batchsize: (idx + 1) * batchsize, :]
            yield r.qid.values, r.rel.values, r[['{}'.format(i) for i in range(1, self.num_features + 1)]].values
            idx += 1

    def build_mislabeled(self, mislabeled_dict, mislabeled_type=None):
        if self.mislabeled_dict is None:
            self.mislabeled_dict = mislabeled_dict
        else:
            self.mislabeled_dict.update(mislabeled_dict)
        self.mislabeled_on = True
        self.mislabeled_type = mislabeled_type

    def get_sampled_qids(self):
        if self.sampled_qids is None:
            with open('model/data/'+self.dataset_type+'/Fold1/sampeled_1000_train_qids.pkl', 'rb') as fp:
                self.sampled_qids = pickle.load(fp)
            if True:
                self.sampled_qids = self.sampled_qids[:100]
            self.num_sessions = len(self.sampled_qids)
        return self.sampled_qids
    
    def get_qids(self):
        if self.qids is not None:
            return
        if self.dataset_type in ['mslr-web30k', 'mslr-web10k'] and self.small_train:
            self.qids = self.get_sampled_qids()
        else:
            if self.df is None:
                self.load()
            self.qids = self.df.qid.unique()
        self.num_sessions = len(self.qids)
    
        
    def indexing_batch_per_query(self, idx, df=None):
        if idx in self.drop_queries:
            return None, None
        if df is None:
            df = self.df
        self.get_qids()
        assert len(self.qids) > idx        
        self.get_cached_batch_per_query(df, self.qids)
        self.current_idx = idx
        qid = self.qids[idx]
        X = self.X[idx][:, :self.num_features]
        Y = self.Y[idx]
        if self.mislabeled_on and (self.mislabeled_dict is not None) and (str(idx) in self.mislabeled_dict.keys()):
            Y = self.mislabeled_dict[str(idx)]
        if self.binary_label:
            Y = (Y > 1).to(torch.int)
        return X, Y            

                
    def get_cached_batch_per_query(self, df, qids):
        if self.dataset_type in ['naver', 'naver_click', 'mq2008-semi']:
            return
        
        if self.dataset_type in ['mslr-web30k', 'mslr-web10k']:
            #print(self.dataset_type, self.data_type, len(qids))
            file_path = os.getcwd()+'/model/data/'+self.dataset_type+'/Fold1/cached_'+self.data_type+'_'+str(len(qids))
            if self.device == 0:
                file_path = file_path+'_cuda0.pkl'
            else:
                file_path = file_path+'.pkl'
        if self.X is not None:
            return
        if os.path.exists(file_path):
            with open(file_path, 'rb') as fp:
                self.X, self.Y = pickle.load(fp)
        else:
            X, Y = [], []
            for qid in tqdm(qids):
                df_qid = df[df.qid == qid]
                x = torch.tensor(df_qid[['{}'.format(i) for i in range(1, self.num_features + 1)]].values).cuda(self.device)
                y = torch.tensor(df_qid.rel.values).cuda(self.device)
                X.append(x)
                Y.append(y)
            with open(file_path, 'wb') as fp:
                pickle.dump((X, Y), fp, pickle.HIGHEST_PROTOCOL)
            self.X = X
            self.Y = Y
        if self.dataset_type in ['mslr-web30k', 'mslr-web10k'] and self.small_train:
            self.X = self.X[:100]
            self.Y = self.Y[:100]
    
    def generate_batch_per_query(self, df=None, shuffle=False):
        """
        :param df: pandas.DataFrame
        :return: X for features, y for relavance
        :rtype: numpy.ndarray, numpy.ndarray
        """
        if df is None:
            df = self.df
        self.get_qids()
        self.get_cached_batch_per_query(df, self.qids)
        
        if True:
            order = list(range(len(self.qids)))
            if shuffle:
                np.random.shuffle(order)
            for i in order:
                if i in self.drop_queries:
                    continue
                self.current_idx = i
                qid = self.qids[i]
                X = self.X[i][:, :self.num_features]
                Y = self.Y[i]
                if self.mislabeled_on and (self.mislabeled_type != 'RAND2') \
                        and (self.mislabeled_dict is not None) and (str(i) in self.mislabeled_dict.keys()):
                    Y = self.mislabeled_dict[str(i)]
#                     print('LABEL CHANGED in idx', i)
#                     print(self.Y[i])
#                     print(self.mislabeled_dict[str(i)])
#                     assert 1 == 2
                if self.binary_label:
                    Y = (Y > 1).to(torch.int)
                yield X, Y            
        else:
            if shuffle:
                np.random.shuffle(self.qids)
            for qid in self.qids:
                df_qid = df[df.qid == qid]
                X = df_qid[['{}'.format(i) for i in range(1, self.num_features + 1)]].values
                #print(X[:10])
                Y = df_qid.rel.values
                if self.mislabeled_on and (self.mislabeled_dict is not None) and (str(i) in self.mislabeled_dict.keys()):
                    Y = self.mislabeled_dict[str(i)]
                yield X, Y

    def load(self):
        """
        :return: pandas.DataFrame
        """
        if os.path.isfile(self.pickle_path):
            print(get_time(), "load from pickle file {}".format(self.pickle_path))
            self.df = pd.read_pickle(self.pickle_path)
            self.num_features = len(self.df.columns) - 2
            self.num_paris = None
            self.num_sessions = len(self.df.qid.unique())
        else:
            self.df = self._parse_feature_and_label(self._load_mslr())
            self.df.to_pickle(self.pickle_path)
        return self.df

    def train_scaler_and_transform(self):
        """Learn a scalar and apply transform."""
        feature_columns = [str(i) for i in range(1, self.num_features + 1)]
        X_train = self.df[feature_columns]
        scaler = preprocessing.StandardScaler().fit(X_train)
        self.df[feature_columns] = scaler.transform(X_train)
        return self.df, scaler

    def apply_scaler(self, scaler):
        print(get_time(), "apply scaler to transform feature for {}".format(self.path))
        feature_columns = [str(i) for i in range(1, self.num_features + 1)]
        X_train = self.df[feature_columns]
        self.df[feature_columns] = scaler.transform(X_train)
        return self.df

    
class NaverLoader(DataLoader):
    def __init__(self, data_type, drop_high_rel):
        self.data_type = data_type
        self.df = None
        self.num_pairs = None
        self.num_sessions = None
        self.drop_high_rel = drop_high_rel
        self.load()
        self.new_vars(dataset_type='naver', data_type=data_type)

    def load(self):
        naver_path = 'model/data/naver/' + self.data_type
        categorical_features = [121,128,230,302,304,306,307,313,314,398,402,403,405,413,414,415,416,417,418]

        with open(naver_path+'.dat', 'r', encoding='utf8') as fp:
            data = fp.read()
            data = data.split('\n')

        df_dict = {'q': [], 'qid': [], 'rel': [], 'url': []}
        
        random.seed(7777)
        nsr = 16 #negative_sampling_ratio
        dict = {}
        for d in data:
            if len(d.split('\t')) < 4:
                print(d)
                continue
            q, url, rel, feature = d.split('\t')
            feature = [f.split(':')[1] for f in feature.split(' ') if int(f.split(':')[0]) not in categorical_features]
            feature = [float(f) if f != 'na' else 0. for f in feature]
            
            if self.drop_high_rel and int(rel) in [4, 5]:
                continue
            
            if self.data_type == 'test' and int(rel) == 1 and random.choice([1]*nsr+[0]*(100-nsr)) == 0:
                continue
            rel = int(rel)-1
            if q in dict.keys():
                dict[q].append((torch.tensor(feature), torch.tensor(rel), url))
            else:
                dict[q]= [(torch.tensor(feature), torch.tensor(rel), url)]
            df_dict['qid'].append(len(list(dict.keys()))-1)
            df_dict['q'].append(q)
            df_dict['rel'].append(rel)
            for i, f in enumerate(feature):
                if str(i+1) not in df_dict.keys():
                    df_dict[str(i+1)] = []
                df_dict[str(i+1)].append(f)
            df_dict['url'].append(url)
            
        X = []
        Y = []
        for q in dict.keys():
            x = None
            y = None
            for _set in dict[q]:
                _x, _y, _ = _set
                _x = _x.view(1, -1)
                _y = _y.view(-1)
                if x is None:
                    x = _x
                    y = _y
                else:
                    x = torch.cat([x, _x], dim=0)
                    y = torch.cat([y, _y], dim=0)
            X.append(x)
            Y.append(y)
            
        self.X = X
        self.Y = Y
        self.df = pd.DataFrame(df_dict)
        print(sum([(y == 0).nonzero().size()[0] for y in Y]) / len(Y))
        self.num_features = self.X[0].size()[-1]
        self.num_paris = None
        self.num_sessions = len(self.df.qid.unique())
        
        
class MQ2008semiLoader(DataLoader):
    def __init__(self, data_type, device):
        self.data_type = data_type
        self.device = device
        self.df = None
        self.num_pairs = None
        self.num_sessions = None
        self.true_labels = []
        self.load()
        self.new_vars(dataset_type='mq2008-semi', data_type=data_type)

    def load(self):
        data_path = 'model/data/MQ2008-semi/' + self.data_type

        with open(data_path+'.txt', 'r', encoding='utf8') as fp:
            data = fp.read()
            data = data.split('\n')

        df_dict = {'qid': [], 'rel': []}

        dict = {}
        for d in tqdm(data):
            d = d.split()
            if len(d) < 2:
                continue
            rel = int(d[0])
            true_rel = int(d[0])
            if rel == -1:
                rel = 0
            q = d[1].split('qid:')[-1]
            feature = [float(v.split(':')[-1]) for v in d[2:-9]]

            if q in dict.keys():
                dict[q].append((feature, rel, true_rel))
            else:
                dict[q]= [(feature, rel, true_rel)]
            df_dict['qid'].append(len(list(dict.keys()))-1)
            df_dict['rel'].append(rel)
            for i, f in enumerate(feature):
                if str(i+1) not in df_dict.keys():
                    df_dict[str(i+1)] = []
                df_dict[str(i+1)].append(f)

        X = []
        Y = []
        for q in tqdm(dict.keys()):
            x = None
            y = None
            ty = []
            for _set in dict[q]:
                _x, _y, _ty = _set
                _x = torch.tensor(_x, dtype=torch.float32, device='cuda:'+str(self.device)).view(1, -1)
                _y = torch.tensor(_y, device='cuda:'+str(self.device)).view(-1)
                ty.append(_ty)
                if x is None:
                    x = _x
                    y = _y
                else:
                    x = torch.cat([x, _x], dim=0)
                    y = torch.cat([y, _y], dim=0)
            X.append(x)
            Y.append(y)
            self.true_labels.append(ty)
            
        self.X = X
        self.Y = Y
        self.df = pd.DataFrame(df_dict)
        print(sum([(y == 0).nonzero().size()[0] for y in Y]) / len(Y))
        self.num_features = self.X[0].size()[-1]
        self.num_pairs = None
        self.num_sessions = len(self.df.qid.unique())
        
        
        
class NaverClickLoader(DataLoader):
    def __init__(self, data_type, device):
        self.data_type = data_type
        self.device = device
        self.df = None
        self.num_pairs = None
        self.num_sessions = None
        self.load()
        self.new_vars(dataset_type='naver_click', data_type=data_type)

    def load(self, filter_unlabeled=True):
        if self.data_type in ['train', 'valid']:
            f_name = 'l2r_data_training_wl.txt'
            valid_size = 913
        elif self.data_type == 'test':
            f_name = 'l2r_data_test.txt'
        naver_click_path = 'model/data/naver_click/' + f_name
        categorical_features = [1,5,6,7,8,10,11,12,13,14,16,19,20,21,29,31,32,33,34,37,38,39,43,44,45,48,49,50,
                                52,53,54,55,56,57,58,59,60,61,62,65,66,67,69,71,73,75,76,77,78,80,81,83,85,86,87,
                                89,90,91,92,94,96,99,100,101,102,103,104,105,107,108,109,111,112,113,114,115,116,
                                118,120,121,122,123,124,126,127,128,129,130,131,133,134,136,138,139,140,143,144,
                                145,146,147,150,151,153,155,158,159,162,163,164,166,167,168,169,170,171,172,173,
                                175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,191,194,196,197,201,
                                202,204,205,206,207,208,210,211,212,213,214,215,216,217,219,220,221,222,224,225,
                                230,234,235,236,237,238,239,240,241,243,244,245,246,247,257,259,260,261,263,264,
                                265,266,267,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,
                                287,288,290,291,292,293,294,295,296,298,299,300,301,302,303,304,305,306,307,313,
                                314,317,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,
                                337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,
                                357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,
                                377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,
                                397,398,402,403,405,407,411,413,414,415,416,417,418,419,420,421,422,423,424,425,
                                426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,
                                446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,
                                466,467,468,469,470,471,472,473,474,475,476,477,478,479,480,481,482,483,484,485,
                                486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,504,505,506,507,508,
                                509,510,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,
                                546,547,548,549,550,595,596,597,598,599,600,601,602,603,604,605,606,607,608,609,
                                610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,
                                630,631,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,649,
                                650,651,652,653,654,655,656,657,658,659,660,661,662,663,664,665,666,667,668,669,
                                670,671,672,673,674,675,676,677,678,679,680,681,682,683,684,685,686,687,688,689,
                                690,691,692,693,694,695,696,697,698,699,700,718,719,720,721,722,723,724,725,726,
                                727,728,729,730,731,732,733,734,735,736,737,738,739,740,741,742,743,744,745,746,
                                747,748,749,750,780,781,782,783,784,785,786,787,788,789,790,791,792,793,794,795,
                                796,797,798,799,800,805,806,807,808,809,810,819,820,821,822,823,824,825,826,827,
                                828,829,830]
        with open(naver_click_path, 'r', encoding='utf8') as fp:
            data = fp.read()
            data = data.split('\n')
        df_dict = {'q': [], 'qid': [], 'rel': [], 'url': []}

        random.seed(7777)
        dict = {}
        for idx in tqdm(range(len(data))):
            d = data[idx]
            if len(d.split('\t')) < 4:
                continue
            q, url, rel, feature = d.split('\t')
            feature = [f.split(':')[1] for f in feature.split(' ') if int(f.split(':')[0]) not in categorical_features]
            feature = [float(f) if f != 'na' else 0. for f in feature]

            if filter_unlabeled and int(rel) == -1:
                continue    
            rel = int(rel)-1
            if self.data_type in ['train', 'valid']:
                if rel < 2:
                    rel = 0
                else:
                    rel = 1
                
            if q in dict.keys():
                dict[q].append((torch.tensor(feature), torch.tensor(rel), url))
            else:
                dict[q]= [(torch.tensor(feature), torch.tensor(rel), url)]
            df_dict['qid'].append(len(list(dict.keys()))-1)
            df_dict['q'].append(q)
            df_dict['rel'].append(rel)
            for i, f in enumerate(feature):
                if str(i+1) not in df_dict.keys():
                    df_dict[str(i+1)] = []
                df_dict[str(i+1)].append(f)
            df_dict['url'].append(url)
        df = pd.DataFrame(df_dict)

        X = []
        Y = []
        for q in dict.keys():
            x = None
            y = None
            for _set in dict[q]:
                _x, _y, _ = _set
                _x = torch.tensor(_x, dtype=torch.float32, device='cuda:'+str(self.device)).view(1, -1)
                _y = torch.tensor(_y, device='cuda:'+str(self.device)).view(-1)
                if x is None:
                    x = _x
                    y = _y
                else:
                    x = torch.cat([x, _x], dim=0)
                    y = torch.cat([y, _y], dim=0)
            X.append(x)
            Y.append(y)

#         print(sum([(y == 0).nonzero().size()[0] for y in Y]) / len(Y))
#         print(sum([(y == 1).nonzero().size()[0] for y in Y]) / len(Y))
#         print(sum([(y == 2).nonzero().size()[0] for y in Y]) / len(Y))
#         print(sum([(y == 3).nonzero().size()[0] for y in Y]) / len(Y))
#         print(sum([(y == 4).nonzero().size()[0] for y in Y]) / len(Y))
        num_features = X[0].size()[-1]
        num_sessions = len(df.qid.unique())
        
        
        df = pd.DataFrame(df_dict)
        if self.data_type in ['train', 'valid']:
            train_size = len(df.qid.unique()) - valid_size
            split_index = df[df['qid'] == (train_size-1)].index[-1]
            if self.data_type == 'train':
                self.df = df.iloc[:split_index+1]
                self.X = X[:train_size]
                self.Y = Y[:train_size]
            elif self.data_type == 'valid':
                self.df = df.iloc[split_index+1:].reset_index(drop=True)
                self.X = X[train_size:]
                self.Y = Y[train_size:]
        else:
            self.X = X
            self.Y = Y
            self.df = df
        self.num_features = self.X[0].size()[-1]
        self.num_paris = None
        self.num_sessions = len(self.df.qid.unique())
        
