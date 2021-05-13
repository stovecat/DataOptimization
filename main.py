from pair_influence import *
from eval import *


def load_init_trainset_and_theta(opt):
        
    #Build data
    train_loader, df_train, valid_loader, df_valid, test_loader, df_test = load_data(device=device, 
                                                                                     dataset_type=opt['dataset_type'],
                                                                                     drop_high_rel=opt['drop_high_rel'])
    if opt['dataset_type'] in ['mslr-web30k', 'mslr-web10k'] and opt['small_train'] == True:
        small_train = True
    else:
        small_train = False
    train_loader.new_vars(small_train=small_train, device=device, data_type='train', dataset_type=opt['dataset_type'])
    valid_loader.new_vars(small_train=False, device=device, data_type='valid', dataset_type=opt['dataset_type'])
    test_loader.new_vars(small_train=False, device=device, data_type='test', dataset_type=opt['dataset_type'])
    
                
    #Change label
    build_mislabeled_dataset(train_loader, error_query_ratio=opt['error_query_ratio'], 
                             error_doc_ratio=opt['error_doc_ratio'], error_type=opt['error_type'])
    
        
    data_dict = {'train_loader': train_loader, 'df_train': df_train, 
                 'valid_loader': valid_loader, 'df_valid': df_valid, 
                 'test_loader': test_loader, 'df_test': df_test,
                'opt': opt['name'], 'dataset_type': opt['dataset_type']}
    
    log_path, csv_path = get_output_path(data_dict)
    
    if os.path.exists(log_path+opt['name']+'.txt'):
        #GET BEST MODEL
        with open(log_path+opt['name']+'.txt', 'r', encoding='utf8') as fp:
            log = fp.read()
        best_epoch = int(log.split(',')[0].split('(')[1])
        test_ndcg_result_log = float(log.split('10: ')[-1].split(',')[0])
        
        model, _ = get_model(train_loader, ckpt_epoch=best_epoch, opt=opt['name'], device=device)
        
        #EVALUATION
        valid_ndcg_result, valid_result = eval_ndcg_at_k(model, device, df_valid, valid_loader)
        test_ndcg_result, test_result = eval_ndcg_at_k(model, device, df_test, test_loader)

        assert test_ndcg_result[10] == test_ndcg_result_log  
        train_loss = get_loss(model, data_dict['train_loader'], criterion, 
                              list(range(0, data_dict['train_loader'].num_sessions)), bar=True)
        train_loss = params_to_tensor(train_loss).sum().item()
        valid_loss = get_loss(model, data_dict['valid_loader'], criterion, 
                              list(range(0, data_dict['valid_loader'].num_sessions)), bar=True)
        valid_loss = params_to_tensor(valid_loss).sum().item()
        test_loss = get_loss(model, data_dict['test_loader'], criterion, 
                              list(range(0, data_dict['test_loader'].num_sessions)), bar=True)
        test_loss = params_to_tensor(test_loss).sum().item()
        print('train loss:', train_loss)
        print('valid loss:', valid_loss)
        print('test  loss:', test_loss)
        
        if 'csv_mode' in opt.keys() or not os.path.exists(csv_path):
            write_log_csv(model, data_dict, device, best_epoch)
    else:
        if 'csv_mode' in opt.keys():
            return None, None
        model = train_theta(data_dict)
        
    clear_seed_all()
    #assert 1 == 2
    return data_dict, model



def get_influences(model, data_dict, unit, refresh=True, q_mean=False):
    #TBD
    #influences = 
    if unit == 'document':
        query_level = False
        pair_level = False
    elif unit == 'query':
        query_level = True
        pair_level = False
    elif unit == 'pair':
        query_level = False
        pair_level = True
    else:
        raise NotImplementedError

    influences = get_influence_on_test_loss(model,
                                            data_dict['train_loader'],
                                            data_dict['valid_loader'],
                                            criterion,
                                            test_indices=list(range(0, data_dict['valid_loader'].num_sessions)),
                                            query_level=query_level,
                                            pair_level=pair_level,
                                            force_refresh=refresh,
                                            device=device,
                                            q_mean=q_mean)
    
    influences_path = 'model/ckptdir/'+data_dict['opt']+'_influences/'
    if not os.path.exists(influences_path):
        os.makedirs(influences_path)
    if query_level:
        influences_path += 'query_influences'
    else:
        influences_path += 'influences'
    influences_path += '.pkl'
    with open(influences_path, 'wb') as fp:
        pickle.dump(influences, fp, pickle.HIGHEST_PROTOCOL)
    
    clear_seed_all()
#     assert 1 == 2

    return influences


def drop_min_n(data_dict, influences, n, unit):
    if unit == 'document':
        #Globally min n을 선택하고 query idx를 key doc idx list를 value로 가지는 dict 생성
        infl_idx = []
        for q_idx, _influences in enumerate(influences):
            for doc_idx, _influence in enumerate(_influences):
                infl_idx.append((_influence, q_idx, doc_idx))
        sorted_infl_idx = sorted(infl_idx, key=lambda x:x[0])[:n]
        drop_documents = {}
        for infl_idx in sorted_infl_idx:
            q_idx = infl_idx[1]
            doc_idx = infl_idx[2]
            if q_idx in drop_documents.keys():
                drop_documents[q_idx].append(doc_idx)
            else:
                drop_documents[q_idx] = [doc_idx]
        
        print('Selected drop documents:', drop_documents)

        for key in drop_documents.keys():
            if key in data_dict['train_loader'].drop_documents:
                data_dict['train_loader'].drop_documents[key].extend(drop_documents[key])
            else:
                data_dict['train_loader'].drop_documents[key] = drop_documents[key]
    elif unit == 'query':
        #Globally min n을 선택하고 query idx를 value로 가지는 list 생성
        infl_idx = [(infl, i) for i, infl in enumerate(influences)]
        sorted_infl_idx = sorted(infl_idx, key=lambda x:x[0])[:n]
        drop_queries = [v[1] for v in sorted_infl_idx]
        
        print('Selected drop queries:', drop_queries)
        
        data_dict['train_loader'].drop_queries.extend(drop_queries)
        if data_dict['train_loader'].mislabeled_dict is not None:
            is_dropped_noise(data_dict['train_loader'].drop_queries, list(data_dict['train_loader'].mislabeled_dict.keys()))
    elif unit == 'pair':
        infl_idx = []
        for q_idx, _influences in enumerate(influences):
            for doc_idx, _influence in enumerate(_influences):
                for doc_idx2, __influence in enumerate(_influence):
                    infl_idx.append((__influence, q_idx, doc_idx, doc_idx2))
        sorted_infl_idx = sorted(infl_idx, key=lambda x:x[0])[:n]
        drop_pairs = {}
        for infl_idx in sorted_infl_idx:
            q_idx = infl_idx[1]
            doc_idx = infl_idx[2]
            doc_idx2 = infl_idx[3]
            if q_idx in drop_pairs.keys():
                drop_pairs[q_idx].append((doc_idx, doc_idx2))
            else:
                drop_pairs[q_idx] = [(doc_idx, doc_idx2)]
        
        print('Selected drop pairs:', drop_pairs)

        for key in drop_pairs.keys():
            if key in data_dict['train_loader'].drop_pairs:
                data_dict['train_loader'].drop_pairs[key].extend(drop_pairs[key])
            else:
                data_dict['train_loader'].drop_pairs[key] = drop_pairs[key]        
        
    clear_seed_all()

    return data_dict

def drop_min_n_with_threshold(data_dict, influences, n, unit, threshold=0.):
    if unit == 'document':
        #Globally min n을 선택하고 query idx를 key doc idx list를 value로 가지는 dict 생성
        infl_idx = []
        for q_idx, _influences in enumerate(influences):
            for doc_idx, _influence in enumerate(_influences):
                if data_dict['dataset_type'] == 'mq2008-semi' \
                    and data_dict['train_loader'].true_labels[q_idx][doc_idx] != -1:
                    continue
                if _influence < threshold:
                    infl_idx.append((_influence, q_idx, doc_idx))
        if infl_idx == []:
            return None
        sorted_infl_idx = sorted(infl_idx, key=lambda x:x[0])[:n]
        drop_documents = {}
        drop_infl = {}
        for infl_idx in sorted_infl_idx:
            tmp_infl = infl_idx[0]
            q_idx = infl_idx[1]
            doc_idx = infl_idx[2]
            if q_idx in drop_documents.keys():
                drop_documents[q_idx].append(doc_idx)
                drop_infl[q_idx].append(tmp_infl)
            else:
                drop_documents[q_idx] = [doc_idx]
                drop_infl[q_idx] = [tmp_infl]

#         with open('IDS_'+data_dict['opt']+'_drop_infl'+'.pkl', 'wb') as fp:
#             pickle.dump(drop_infl, fp, pickle.HIGHEST_PROTOCOL)

        
        print('Selected drop documents:', drop_documents)
        if data_dict['dataset_type'] == 'mq2008-semi':
            for q_idx in drop_documents.keys():
                for doc_idx in drop_documents[q_idx]:
                    assert data_dict['train_loader'].true_labels[q_idx][doc_idx] == -1

        for key in drop_documents.keys():
            if key in data_dict['train_loader'].drop_documents:
                data_dict['train_loader'].drop_documents[key].extend(drop_documents[key])
            else:
                data_dict['train_loader'].drop_documents[key] = drop_documents[key]
    elif unit == 'query':
        #Globally min n을 선택하고 query idx를 value로 가지는 list 생성
        infl_idx = [(infl, i) for i, infl in enumerate(influences) if infl < threshold]
        if infl_idx == []:
            return None
        sorted_infl_idx = sorted(infl_idx, key=lambda x:x[0])[:n]
        drop_queries = [v[1] for v in sorted_infl_idx]
        
        print('Selected drop queries:', drop_queries)
        
        data_dict['train_loader'].drop_queries.extend(drop_queries)
        if data_dict['train_loader'].mislabeled_dict is not None:
            is_dropped_noise(data_dict['train_loader'].drop_queries, list(data_dict['train_loader'].mislabeled_dict.keys()))
    elif unit == 'pair':
        infl_idx = []
        for q_idx, _influences in enumerate(influences):
            for doc_idx, _influence in enumerate(_influences):
                for doc_idx2, __influence in enumerate(_influence):
                    if __influence < threshold:
                        infl_idx.append((__influence, q_idx, doc_idx, doc_idx2))
        if infl_idx == []:
            return None
        sorted_infl_idx = sorted(infl_idx, key=lambda x:x[0])[:n]
        drop_pairs = {}
        drop_infl = {}
        for infl_idx in sorted_infl_idx:
            tmp_infl = infl_idx[0]
            q_idx = infl_idx[1]
            doc_idx = infl_idx[2]
            doc_idx2 = infl_idx[3]
            if q_idx in drop_pairs.keys():
                drop_pairs[q_idx].append((doc_idx, doc_idx2))
                drop_infl[q_idx].append(tmp_infl)
            else:
                drop_pairs[q_idx] = [(doc_idx, doc_idx2)]
                drop_infl[q_idx] = [tmp_infl]
        
        with open('IPS_'+data_dict['opt']+'_drop_infl_original'+'.pkl', 'wb') as fp:
            pickle.dump(drop_infl, fp, pickle.HIGHEST_PROTOCOL)
            
        print('Selected drop pairs:', drop_pairs)

        for key in drop_pairs.keys():
            if key in data_dict['train_loader'].drop_pairs:
                data_dict['train_loader'].drop_pairs[key].extend(drop_pairs[key])
            else:
                data_dict['train_loader'].drop_pairs[key] = drop_pairs[key]
                
#     with open('IDS_'+data_dict['opt']+'_dropped_pair_original.pkl', 'wb') as fp:
#         if unit == 'document':
#             dropped = data_dict['train_loader'].drop_documents
#         elif unit == 'pair':
#             dropped = data_dict['train_loader'].drop_pairs
#         pickle.dump(dropped, fp, pickle.HIGHEST_PROTOCOL)
    #assert 1 == 2

                        
    clear_seed_all()

    return data_dict


def drop_random_n(model, data_dict, n, unit):
    indices = list(range(data_dict['train_loader'].num_sessions))
    losses = []
    if unit == 'document':
        for idx in tqdm(indices):
            tmp_loss = get_doc_loss(model, data_dict['train_loader'], criterion, [idx])
            if tmp_loss == []:
                losses.append(torch.tensor(0.).cpu())
            else:
                losses.append(tmp_loss[0].detach().cpu())
        #Randomly n을 선택하고 query idx를 key doc idx list를 value로 가지는 dict 생성
        loss_idx = []
        for q_idx, _losses in enumerate(losses):
            if _losses.size() == torch.Size([]):
                continue
            for doc_idx, _loss in enumerate(_losses):
                if q_idx in data_dict['train_loader'].drop_documents.keys() \
                                        and doc_idx in data_dict['train_loader'].drop_documents[q_idx]:
                    continue
                if _loss != 0.:
                    loss_idx.append((q_idx, doc_idx))
        random.shuffle(loss_idx)
        random.shuffle(loss_idx)
        shuffled_loss_idx = loss_idx[:n]
        drop_documents = {}
        for _loss_idx in shuffled_loss_idx:
            q_idx = _loss_idx[0]
            doc_idx = _loss_idx[1]
            if q_idx in drop_documents.keys():
                drop_documents[q_idx].append(doc_idx)
            else:
                drop_documents[q_idx] = [doc_idx]
        
        print('Selected drop documents:', drop_documents)
        
        for key in drop_documents.keys():
            if key in data_dict['train_loader'].drop_documents:
                data_dict['train_loader'].drop_documents[key].extend(drop_documents[key])
            else:
                data_dict['train_loader'].drop_documents[key] = drop_documents[key]
        #is_dropped_noise(data_dict['train_loader'].drop_documents, data_dict['train_loader'].mislabeled)
    elif unit == 'query':
        for idx in tqdm(indices):
            tmp_loss = get_query_loss(model, data_dict['train_loader'], criterion, [idx])
            if tmp_loss == []:
                losses.append(torch.tensor(0.).cpu())
            else:
                losses.append(tmp_loss[0].detach().cpu())
        loss_idx = [i for i, loss in enumerate(losses) if (loss != 0.).int().sum() > 0.]
        random.shuffle(loss_idx)
        random.shuffle(loss_idx)
        shuffled_loss_idx = loss_idx[:n]
        drop_queries = shuffled_loss_idx
        
        print('Selected drop queries:', drop_queries)
        
        data_dict['train_loader'].drop_queries.extend(drop_queries)
        if data_dict['train_loader'].mislabeled_dict is not None:
            is_dropped_noise(data_dict['train_loader'].drop_queries, list(data_dict['train_loader'].mislabeled_dict.keys()))

    clear_seed_all()

    return data_dict


def drop_oracle_n(model, data_dict, n, unit):
    torch.cuda.set_device(device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    
    if data_dict['train_loader'].mislabeled_dict is None:
        clear_seed_all()
        return data_dict
        
    if unit == 'document':
        #mislabeled_dict에서 n개만큼 drop
        cand_idx = []
        if data_dict['train_loader'].mislabeled_type == 'RAND2':
            #TBD
            raise NotImplementedError
        
        for q_idx in data_dict['train_loader'].mislabeled_dict.keys():
            for doc_idx in data_dict['train_loader'].mislabeled_dict[q_idx]:
                if q_idx in data_dict['train_loader'].drop_documents.keys() \
                                        and doc_idx in data_dict['train_loader'].drop_documents[q_idx]:
                    continue
                cand_idx.append((q_idx, doc_idx))
        
        random.shuffle(cand_idx)
        random.shuffle(cand_idx)
        shuffled_cand_idx = cand_idx[:n]
        drop_documents = {}
        for _cand_idx in shuffled_cand_idx:
            q_idx = _cand_idx[0]
            doc_idx = _cand_idx[1]
            if q_idx in drop_documents.keys():
                drop_documents[q_idx].append(doc_idx)
            else:
                drop_documents[q_idx] = [doc_idx]
        
        print('Selected drop documents:', drop_documents)
        
        for key in drop_documents.keys():
            if key in data_dict['train_loader'].drop_documents:
                data_dict['train_loader'].drop_documents[key].extend(drop_documents[key])
            else:
                data_dict['train_loader'].drop_documents[key] = drop_documents[key]
        #is_dropped_noise(data_dict['train_loader'].drop_documents, data_dict['train_loader'].mislabeled)
    elif unit == 'query':
        #Globally min n을 선택하고 query idx를 value로 가지는 list 생성
        cand_idx = [int(idx) for idx in data_dict['train_loader'].mislabeled_dict.keys() \
                    if int(idx) not in data_dict['train_loader'].drop_queries]
        random.shuffle(cand_idx)
        random.shuffle(cand_idx)
        shuffled_cand_idx = cand_idx[:n]
        drop_queries = shuffled_cand_idx
        
        print('Selected drop queries:', drop_queries)
        
        data_dict['train_loader'].drop_queries.extend(drop_queries)
        if data_dict['train_loader'].mislabeled_dict is not None:
            is_dropped_noise(data_dict['train_loader'].drop_queries, list(data_dict['train_loader'].mislabeled_dict.keys()))

    clear_seed_all()

    return data_dict


def drop_high_rel(data_dict):
    torch.cuda.set_device(device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    
    drop_documents = {}
    for q_idx, y in enumerate(data_dict['train_loader'].Y):
        for doc_idx, _y in enumerate(y):
            if _y in [3, 4]:
                if q_idx in drop_documents.keys():
                    drop_documents[q_idx].append(doc_idx)
                else:
                    drop_documents[q_idx] = [doc_idx]

    for key in drop_documents.keys():
        if key in data_dict['train_loader'].drop_documents:
            data_dict['train_loader'].drop_documents[key].extend(drop_documents[key])
        else:
            data_dict['train_loader'].drop_documents[key] = drop_documents[key]


    clear_seed_all()

    return data_dict


def drop_by_threshold(data_dict, influences, unit, threshold=0.):
    #TBD
    if unit == 'document':
        #Globally min n을 선택하고 query idx를 key doc idx list를 value로 가지는 dict 생성
        infl_idx = []
        for q_idx, _influences in enumerate(influences):
            for doc_idx, _influence in enumerate(_influences):
                if _influence < threshold:
                    infl_idx.append((_influence, q_idx, doc_idx))
                    
        sorted_infl_idx = infl_idx
        drop_documents = {}
        for infl_idx in sorted_infl_idx:
            q_idx = infl_idx[1]
            doc_idx = infl_idx[2]
            if q_idx in drop_documents.keys():
                drop_documents[q_idx].append(doc_idx)
            else:
                drop_documents[q_idx] = [doc_idx]
        
        print('Selected drop documents:', drop_documents)

        for key in drop_documents.keys():
            if key in data_dict['train_loader'].drop_documents:
                data_dict['train_loader'].drop_documents[key].extend(drop_documents[key])
            else:
                data_dict['train_loader'].drop_documents[key] = drop_documents[key]
    elif unit == 'query':
        #Globally min n을 선택하고 query idx를 value로 가지는 list 생성
        infl_idx = [(infl, i) for i, infl in enumerate(influences) if infl < threshold]
        sorted_infl_idx = infl_idx
        drop_queries = [v[1] for v in sorted_infl_idx]
        
        print('Selected drop queries:', drop_queries)
        
        data_dict['train_loader'].drop_queries.extend(drop_queries)
        if data_dict['train_loader'].mislabeled_dict is not None:
            is_dropped_noise(data_dict['train_loader'].drop_queries, list(data_dict['train_loader'].mislabeled_dict.keys()))
    elif unit == 'pair':
        infl_idx = []
        for q_idx, _influences in enumerate(influences):
            for doc_idx, _influence in enumerate(_influences):
                for doc_idx2, __influence in enumerate(_influence):
                    if __influence < threshold:
                        infl_idx.append((__influence, q_idx, doc_idx, doc_idx2))
        if infl_idx == []:
            return None
        sorted_infl_idx = infl_idx
        drop_pairs = {}
        drop_infl = {}
        for infl_idx in sorted_infl_idx:
            tmp_infl = infl_idx[0]
            q_idx = infl_idx[1]
            doc_idx = infl_idx[2]
            doc_idx2 = infl_idx[3]
            if q_idx in drop_pairs.keys():
                drop_pairs[q_idx].append((doc_idx, doc_idx2))
                drop_infl[q_idx].append(tmp_infl)
            else:
                drop_pairs[q_idx] = [(doc_idx, doc_idx2)]
                drop_infl[q_idx] = [tmp_infl]

        assert 1 == 2

        with open('TPS_'+data_dict['opt']+'_drop_infl'+'.pkl', 'wb') as fp:
            pickle.dump(drop_infl, fp, pickle.HIGHEST_PROTOCOL)
        print('Selected drop pairs:', drop_pairs)

        for key in drop_pairs.keys():
            if key in data_dict['train_loader'].drop_pairs:
                data_dict['train_loader'].drop_pairs[key].extend(drop_pairs[key])
            else:
                data_dict['train_loader'].drop_pairs[key] = drop_pairs[key]   
    clear_seed_all()
    with open(data_dict['opt']+'.pkl', 'wb') as fp:
        if unit == 'document':
            dropped = data_dict['train_loader'].drop_documents
        elif unit == 'pair':
            dropped = data_dict['train_loader'].drop_pairs
        pickle.dump(dropped, fp, pickle.HIGHEST_PROTOCOL)
    #assert 1 == 2


    return data_dict


def is_dropped_noise(dropped_qid, noise_qid):
    total_dropped = len(dropped_qid)
    total_noise = len(noise_qid)
    correct_num = 0
    for d_qid in dropped_qid:
        if str(d_qid) in noise_qid:
            correct_num += 1
    print('total dropped:', total_dropped)
    print('total noise:', total_noise)
    precision = correct_num / total_dropped if total_dropped > 0  else 0.
    recall = correct_num / total_noise if total_noise > 0  else 0.
    print('precision:', precision)
    print('recall:', recall)
    return total_dropped, total_noise, correct_num


# def is_dropped_noise(dropped_dict, noise_dict):
#     noise_keys = noise_dict.keys()
#     total_dropped = [len(dropped_dict[k]) for k in dropped_dict.keys()]
#     total_noise = [len(noise_dict[k]) for k in noise_dict.keys()]
#     correct_num = 0
#     for dropped_key in dropped_dict.keys():
#         if dropped_key in noise_keys:
#             for doc_idx in dropped_dict[dropped_key]:
#                 if doc_idx in noise_dict[dropped_key]:
#                     correct_num += 1
#     print('total dropped:', total_dropped)
#     print('total noise:', total_noise)
#     print('precision:', correct_num / total_dropped)
#     print('recall:', correct_num / total_noise)


def get_output_path(data_dict):
    if data_dict['dataset_type'] == 'naver':
        log_path = 'log/naver/'
        csv_path = 'csv/naver/'
    elif data_dict['dataset_type'] == 'mq2008-semi':
        log_path = 'log/mq2008-semi/'
        csv_path = 'csv/mq2008-semi/'
    elif data_dict['dataset_type'] == 'naver_click':
        log_path = 'log/naver_click/'
        csv_path = 'csv/naver_click/'
    else:
        log_path = 'log/sigmoid/'
        csv_path = 'csv/sigmoid/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    if "_v" in data_dict['opt']:
        csv_fn = "_v".join(data_dict['opt'].split('_v')[:-1])+'.csv'
    else:
        csv_fn = data_dict['opt']+'.csv'

    return log_path, csv_path+csv_fn


def write_log_csv(model, data_dict, device, best_epoch):
    valid_ndcg_result, valid_result = eval_ndcg_at_k(model, 
                                                   device, 
                                                   data_dict['df_valid'], 
                                                   data_dict['valid_loader'])
    test_ndcg_result, test_result = eval_ndcg_at_k(model, 
                                                   device, 
                                                   data_dict['df_test'], 
                                                   data_dict['test_loader'])
    
    def get_result_dict():
        k_for_precision = [1, 3, 5, 10, 30]
        k_for_ndcg = [1, 3, 5, 10, 30]
        result_dict = {'train_loss': []}#{'drop method': [], 'iter': []}
        for set_name in ['valid', 'test']:
            for k in k_for_precision:
                result_dict[set_name+'_P@'+str(k)] = []
            for k in k_for_ndcg:
                result_dict[set_name+'_NDCG@'+str(k)] = []
            result_dict[set_name+'_MAP'] = []
            result_dict[set_name+'_MRR'] = []
            result_dict[set_name+'_loss'] = []
        return result_dict
    
    if data_dict['dataset_type'] in ['naver']:
        min_pos_label = 3
    elif data_dict['dataset_type'] == 'mq2008-semi':
        min_pos_label = 1
    elif data_dict['dataset_type'] == 'naver_click':
        min_pos_label = 1
    else:
        min_pos_label = 2

    
    #EVALUATION
    valid_ndcg_result, valid_result = eval_ndcg_at_k(model, device, data_dict['df_valid'], 
                                                     data_dict['valid_loader'], k_list=[1, 3, 5, 10, 30])
    valid_result_dict = evaluation(valid_result[0], valid_result[1], min_pos_label=min_pos_label)
    if data_dict['dataset_type'] == 'naver_click':
        min_pos_label = 3
    test_ndcg_result, test_result = eval_ndcg_at_k(model, device, data_dict['df_test'], 
                                                   data_dict['test_loader'], k_list=[1, 3, 5, 10, 30])
    test_result_dict = evaluation(test_result[0], test_result[1], min_pos_label=min_pos_label)

    
    train_loss = get_loss(model, data_dict['train_loader'], criterion, 
                          list(range(0, data_dict['train_loader'].num_sessions)), bar=True)
    train_loss = params_to_tensor(train_loss).sum().item()
    valid_loss = get_loss(model, data_dict['valid_loader'], criterion, 
                          list(range(0, data_dict['valid_loader'].num_sessions)), bar=True)
    valid_loss = params_to_tensor(valid_loss).sum().item()
    test_loss = get_loss(model, data_dict['test_loader'], criterion, 
                          list(range(0, data_dict['test_loader'].num_sessions)), bar=True)
    test_loss = params_to_tensor(test_loss).sum().item()
    
    tmp_rd = get_result_dict()
#     tmp_rd['drop method'] = opt_path
#     tmp_rd['iter'] = i
    tmp_rd['train_loss'].append(train_loss)
    tmp_rd['valid_P@1'].append(valid_result_dict['P@1'])
    tmp_rd['valid_P@3'].append(valid_result_dict['P@3'])
    tmp_rd['valid_P@5'].append(valid_result_dict['P@5'])
    tmp_rd['valid_P@10'].append(valid_result_dict['P@10'])
    tmp_rd['valid_P@30'].append(valid_result_dict['P@30'])
    tmp_rd['valid_NDCG@1'].append(valid_ndcg_result[1])
    tmp_rd['valid_NDCG@3'].append(valid_ndcg_result[3])
    tmp_rd['valid_NDCG@5'].append(valid_ndcg_result[5])
    tmp_rd['valid_NDCG@10'].append(valid_ndcg_result[10])
    tmp_rd['valid_NDCG@30'].append(valid_ndcg_result[30])
    tmp_rd['valid_MAP'].append(valid_result_dict['MAP'])
    tmp_rd['valid_MRR'].append(valid_result_dict['MRR'])
    tmp_rd['valid_loss'].append(valid_loss)
    tmp_rd['test_P@1'].append(test_result_dict['P@1'])
    tmp_rd['test_P@3'].append(test_result_dict['P@3'])
    tmp_rd['test_P@5'].append(test_result_dict['P@5'])
    tmp_rd['test_P@10'].append(test_result_dict['P@10'])
    tmp_rd['test_P@30'].append(test_result_dict['P@30'])
    tmp_rd['test_NDCG@1'].append(test_ndcg_result[1])
    tmp_rd['test_NDCG@3'].append(test_ndcg_result[3])
    tmp_rd['test_NDCG@5'].append(test_ndcg_result[5])
    tmp_rd['test_NDCG@10'].append(test_ndcg_result[10])
    tmp_rd['test_NDCG@30'].append(test_ndcg_result[30])
    tmp_rd['test_MAP'].append(test_result_dict['MAP'])
    tmp_rd['test_MRR'].append(test_result_dict['MRR'])
    tmp_rd['test_loss'].append(test_loss)
    
    tmp_df = pd.DataFrame(tmp_rd)
    
    
    log_path, csv_path = get_output_path(data_dict)
            
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, tmp_df], ignore_index=True)
    else:
        df = tmp_df
    print(csv_path)
    df.to_csv(csv_path)
    
    if data_dict['train_loader'].mislabeled_dict is not None:
        total_dropped, total_noise, correct_num = is_dropped_noise(data_dict['train_loader'].drop_queries, 
                                                                   list(data_dict['train_loader'].mislabeled_dict.keys()))
    else:
        total_dropped = len(data_dict['train_loader'].drop_queries)
        total_noise = 0
        correct_num = 0

    
    with open(log_path+data_dict['opt']+'.txt', 'w', encoding='utf8') as fp:
        fp.write(str((best_epoch, valid_ndcg_result, test_ndcg_result, 
                      total_dropped, total_noise, correct_num, 
                      train_loss, valid_loss, test_loss)))
        

def train_theta(data_dict):    
    #TRAIN
    best_ndcg_result, best_epoch = train_rank_net(
        data_dict['train_loader'], data_dict['valid_loader'], data_dict['df_valid'],
        args['start_epoch'], args['additional_epoch'], args['lr'], args['optim'],
        args['train_algo'],
        args['double_precision'], args['standardize'],
        args['small_dataset'], args['debug'],
        output_dir=args['output_dir'],
        opt=data_dict['opt'],
        device=device,
        seed=seed
    )

    #GET BEST MODEL
    model, _ = get_model(data_dict['train_loader'], ckpt_epoch=best_epoch, opt=data_dict['opt'], device=device)
    
    write_log_csv(model, data_dict, device, best_epoch)

    return model


def Algorithm(trainset_opt={'error_query_ratio': 0, 
                            'error_doc_ratio': 0, 
                            'error_type': 'RAND', 
                            'name': '0_0_RAND2', 
                            'dataset_type': 'mslr-web30k'}, 
              n=1, num_of_iter=10, unit='document'):
    #[변인] SET: clean / noisy, n: 1 ~ 적당히 큰 수?, 단위: document / query
    print('INFLDROP')
    data_dict, theta = load_init_trainset_and_theta(trainset_opt)
    for i in range(num_of_iter):
        if i > 200:
            break
#         data_dict['opt'] = trainset_opt['dataset_type']+'_'+trainset_opt['name']+ \
#                         '_QUERYMEAN'+'_DROP_'+str(n)+'_'+unit+'_v'+str(i+1)
        data_dict['opt'] = trainset_opt['dataset_type']+'_'+trainset_opt['name']+ \
                        '_RINFLDROP_'+str(n)+'_'+unit+'_v'+str(i+1)

        influences = get_influences(theta, data_dict, unit, refresh=True, q_mean=False)
        data_dict = drop_min_n(data_dict, influences, n, unit)
        theta = train_theta(data_dict)

        
def Algorithm_QMEAN(trainset_opt={'error_query_ratio': 0, 
                            'error_doc_ratio': 0, 
                            'error_type': 'RAND', 
                            'name': '0_0_RAND2', 
                            'dataset_type': 'mslr-web30k', 'drop_high_rel': False}, 
              n=1, num_of_iter=10, unit='document'):
    #[변인] SET: clean / noisy, n: 1 ~ 적당히 큰 수?, 단위: document / query
    data_dict, theta = load_init_trainset_and_theta(trainset_opt)
    for i in range(num_of_iter):
        data_dict['opt'] = trainset_opt['dataset_type']+'_'+trainset_opt['name']+ \
                        '_RQUERYMEAN'+'_DROP_'+str(n)+'_'+unit+'_v'+str(i+1)

        influences = get_influences(theta, data_dict, unit, refresh=True, q_mean=True)
        data_dict = drop_min_n(data_dict, influences, n, unit)
        theta = train_theta(data_dict)
        
        
def Algorithm_RAND(trainset_opt={'error_query_ratio': 0, 
                            'error_doc_ratio': 0, 
                            'error_type': 'RAND', 
                            'name': '0_0_RAND2',
                            'dataset_type': 'mslr-web30k', 'drop_high_rel': False}, 
              n=1, num_of_iter=10, unit='document'):
    #[변인] SET: clean / noisy, n: 1 ~ 적당히 큰 수?, 단위: document / query
    data_dict, theta = load_init_trainset_and_theta(trainset_opt)
    for i in range(num_of_iter):
        data_dict['opt'] = trainset_opt['dataset_type']+'_'+trainset_opt['name']+'_RANDOMDROP_'+str(n)+'_'+unit+'_v'+str(i+1)
        data_dict = drop_random_n(theta, data_dict, n, unit)
        theta = train_theta(data_dict)


def Algorithm_ORACLE(trainset_opt={'error_query_ratio': 0, 
                            'error_doc_ratio': 0, 
                            'error_type': 'RAND', 
                            'name': '0_0_RAND2',
                            'dataset_type': 'mslr-web30k', 'drop_high_rel': False}, 
              n=1, num_of_iter=10, unit='document'):
    #[변인] SET: clean / noisy, n: 1 ~ 적당히 큰 수?, 단위: document / query
    data_dict, theta = load_init_trainset_and_theta(trainset_opt)
    for i in range(num_of_iter):
        data_dict['opt'] = trainset_opt['dataset_type']+'_'+trainset_opt['name']+'_ORACLEDROP_'+str(n)+'_'+unit+'_v'+str(i+1)
        data_dict = drop_oracle_n(theta, data_dict, n, unit)
        theta = train_theta(data_dict)
        
def Algorithm_THRESHOLD(trainset_opt={'error_query_ratio': 0, 
                            'error_doc_ratio': 0, 
                            'error_type': 'RAND', 
                            'name': '0_0_RAND2', 
                            'dataset_type': 'mslr-web30k'}, 
                        unit='document'):
    #[변인] SET: clean / noisy, n: 1 ~ 적당히 큰 수?, 단위: document / query
    data_dict, theta = load_init_trainset_and_theta(trainset_opt)
    if trainset_opt['qmean']:
        data_dict['opt'] = trainset_opt['dataset_type']+'_'+trainset_opt['name']+ \
                        '_QUERYMEAN'+'_THRSHDDROP_'+unit
    else:
        data_dict['opt'] = trainset_opt['dataset_type']+'_'+trainset_opt['name']+ \
                        '_RINFL_THRSHDDROP_'+unit
#     data_dict['opt'] = trainset_opt['dataset_type']+'_'+trainset_opt['name']+ \
#                     '_THRSHDDROP_'+unit
    influences = get_influences(theta, data_dict, unit, refresh=True, q_mean=trainset_opt['qmean'])
    data_dict = drop_by_threshold(data_dict, influences, unit, threshold=0.)
    theta = train_theta(data_dict)
        
def Algorithm_min_THRESHOLD(trainset_opt={'error_query_ratio': 0, 
                            'error_doc_ratio': 0, 
                            'error_type': 'RAND', 
                            'name': '0_0_RAND2', 
                            'dataset_type': 'mslr-web30k'}, 
              n=1, num_of_iter=10, unit='document'):
    #[변인] SET: clean / noisy, n: 1 ~ 적당히 큰 수?, 단위: document / query
    if trainset_opt['qmean']:
        print('QMEANDROP')
    else:
        print('INFLDROP')
    data_dict, theta = load_init_trainset_and_theta(trainset_opt)
    i = 0
    while True:
        if trainset_opt['qmean']:
            data_dict['opt'] = trainset_opt['dataset_type']+'_'+trainset_opt['name']+ \
                            '_QUERYMEAN'+'_DROP_'+str(n)+'_'+unit+'_v'+str(i+1)
        else:
            data_dict['opt'] = trainset_opt['dataset_type']+'_'+trainset_opt['name']+ \
                            '_RINFLDROP_'+str(n)+'_'+unit+'_v'+str(i+1)
        influences = get_influences(theta, data_dict, unit, refresh=True, q_mean=trainset_opt['qmean'])
        data_dict = drop_min_n_with_threshold(data_dict, influences, n, unit, trainset_opt['threshold'])
        if data_dict is None:
            break
        theta = train_theta(data_dict)
        i += 1
        
def Algorithm_CSV(trainset_opt={'error_query_ratio': 0, 
                            'error_doc_ratio': 0, 
                            'error_type': 'RAND', 
                            'name': '0_0_RAND2', 
                            'dataset_type': 'mslr-web30k'}, 
              n=1, num_of_iter=10, unit='document'):
    print('GET CSV')
    i = 0
    name = trainset_opt['name']
    trainset_opt['csv_mode'] = True
    while True:
        if trainset_opt['qmean']:
            trainset_opt['name'] = trainset_opt['dataset_type']+'_'+name+ \
                            '_QUERYMEAN'+'_DROP_'+str(n)+'_'+unit+'_v'+str(i+1)
        else:
            trainset_opt['name'] = trainset_opt['dataset_type']+'_'+name+ \
                            '_RINFLDROP_'+str(n)+'_'+unit+'_v'+str(i+1)
        data_dict, theta = load_init_trainset_and_theta(trainset_opt)
        if data_dict is None:
            break
        i += 1

        
def get_dropped(method):    
    with open(method+'_pair.pkl', 'rb') as fp:
        dropped = pickle.load(fp)
        
    #DANGER: static load
    with open(method+'_infl.pkl', 'rb') as fp:
        dropped_infl = pickle.load(fp)


    return dropped, dropped_infl

def Algorithm_LOO(trainset_opt={'error_query_ratio': 0, 
                            'error_doc_ratio': 0, 
                            'error_type': 'RAND', 
                            'name': '0_0_RAND2', 
                            'dataset_type': 'mslr-web30k'}, 
              n=1, num_of_iter=10, unit='document'):
    method = trainset_opt['method']
    dropped, dropped_infl = get_dropped(method)
    
    
    cnt = 0
    qids = list(dropped.keys())
    clear_seed_all()
    total_dropped = [len(dropped[qid]) for qid in qids]
    already_dropped = []
    infl_list = []
    while len(already_dropped) != sum(total_dropped):
        qids_idx = list(range(0, len(qids)))
        random.shuffle(qids_idx)
        qid = qids[qids_idx[0]]
        
        units = dropped[qid]
        units_idx = list(range(0, len(units)))
        random.shuffle(units_idx)
        unit = units[units_idx[0]]
        if (qid, unit) in already_dropped:
            continue
        infl = dropped_infl[qid][units_idx[0]]
        
        data_dict, _ = load_init_trainset_and_theta(trainset_opt)
        data_dict['opt'] = trainset_opt['dataset_type']+'_'+trainset_opt['name']+ \
                                '_LOO2_'+method+'_v'+str(cnt)
        if type(unit) == int:
            data_dict['train_loader'].drop_documents[qid] = [unit]
        elif type(unit) == tuple:
            data_dict['train_loader'].drop_pairs[qid] = [unit]
        theta = train_theta(data_dict)
        cnt += 1
        already_dropped.append((qid, unit))
        infl_list.append(infl)
        print(already_dropped)
        print(infl_list)
        with open('LOO2_'+method+'_infl.pkl', 'wb') as fp:
            pickle.dump(infl_list, fp, pickle.HIGHEST_PROTOCOL)            
#         if cnt == 100:
#             return

def Algorithm_LGO(trainset_opt={'error_query_ratio': 0, 
                            'error_doc_ratio': 0, 
                            'error_type': 'RAND', 
                            'name': '0_0_RAND2', 
                            'dataset_type': 'mslr-web30k'}, 
              n=1, num_of_iter=10, unit='document'):
    method = trainset_opt['method']
    
    with open(method+'.pkl', 'rb') as fp:
        dropped, dropped_infl = pickle.load(fp)
        
    cnt = 0
    clear_seed_all()
    infl_list = []
    
    for d, infl in zip(dropped, dropped_infl):
        data_dict, _ = load_init_trainset_and_theta(trainset_opt)
        data_dict['opt'] = trainset_opt['dataset_type']+'_'+trainset_opt['name']+ \
                                '_LOO2_'+method+'_v'+str(cnt)
        
        data_dict['train_loader'].drop_pairs = d
        theta = train_theta(data_dict)
        cnt += 1
        infl_list.append(infl)
        print(infl_list)
        with open('LOO2_'+method+'_infl.pkl', 'wb') as fp:
            pickle.dump(infl_list, fp, pickle.HIGHEST_PROTOCOL)       
    assert 1 == 2


seed = 7777
if __name__ == '__main__':
    from tqdm import tqdm
    import sys
    unit = sys.argv[1]
    device = int(sys.argv[2])
    num_of_iter = 0
    dataset_type = 'mslr-web30k'#'mq2008-semi'##'naver_click'#'naver'#'mslr-web30k'#'mslr-web10k'#
    LOO = False#True

    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    is_doc, is_q, is_pair = False, False, False
    assert (('document' in unit) or ('pair' in unit)) != ('query' in unit)
    if 'document' in unit:
        is_doc = True
        #n_list = [3, 100, 1000]
        n_list = [100]
    if 'query' in unit:
        is_q = True
        n_list = [1]
    elif 'pair' in unit:
        is_pair = True
        n_list = [1000]#[100]
        
    for n in tqdm(n_list):
        if len(unit.split('noise')) > 1:
            if len(sys.argv) > 5:
                eqr = int(sys.argv[3])
                edr = int(sys.argv[4])
                et = sys.argv[5]
                name = str(eqr)+'_'+str(edr)+'_'+et
                if et == 'CE2':
                    name += 'v3'
                if len(sys.argv) > 6:
                    #global seed
                    seed = int(sys.argv[6])
            else:
                raise NotImplementedError
            trainset_opt={'error_query_ratio': eqr, 
                            'error_doc_ratio': edr, 
                            'error_type': et, 
                            'name': name, 'drop_high_rel': False,
                            'qmean': False}

        else:
            trainset_opt={'error_query_ratio': 0, 
                                        'error_doc_ratio': 0, 
                                        'error_type': 'RAND2', 
                                        'name': '0_0_RAND2', 'drop_high_rel': False,
                                        'qmean': False}
        original_name = trainset_opt['name']
        if dataset_type is None:
            trainset_opt['dataset_type'] = 'mslr-web30k'
        else:
            trainset_opt['dataset_type'] = dataset_type
        if len(unit.split('high')) > 1:
            trainset_opt['drop_high_rel'] = True
            trainset_opt['name']+='high2'
        if len(unit.split('small')) > 1:
            trainset_opt['name']+='small'
        if len(unit.split('full')) > 1:
            trainset_opt['small_train'] = False
            trainset_opt['name']+='full'
        else:
            trainset_opt['small_train'] = True
            trainset_opt['name']+='small'
        if len(unit.split('lambda')) > 1:
            trainset_opt['name']+='lambda'
        
        #LOO
        if LOO:
            trainset_opt['method'] = 'RGS_'+original_name
            Algorithm_LGO(trainset_opt=trainset_opt, n=n, num_of_iter=num_of_iter, unit=unit.split('_')[0])
            trainset_opt['method'] = 'IDS_'+original_name
            Algorithm_LOO(trainset_opt=trainset_opt, n=n, num_of_iter=num_of_iter, unit=unit.split('_')[0])

        
        if len(unit.split('withthreshold')) > 1:
            if len(unit.split('document')) > 1:
                trainset_opt['name']+='sum'
            if len(unit.split('query')) > 1:
                trainset_opt['name']+='mean'
            trainset_opt['threshold'] = 0.
            trainset_opt['name']+='withthreshold'
            if len(unit.split('qmean')) > 1:
                trainset_opt['qmean'] = True
            if len(unit.split('csv')) > 1:
                Algorithm_CSV(trainset_opt=trainset_opt, n=n, num_of_iter=num_of_iter, unit=unit.split('_')[0])
            else:
                Algorithm_min_THRESHOLD(trainset_opt=trainset_opt, n=n, num_of_iter=num_of_iter, unit=unit.split('_')[0])
        elif len(unit.split('threshold')) > 1:
            if len(unit.split('qmean')) > 1:
                trainset_opt['qmean'] = True
            Algorithm_THRESHOLD(trainset_opt=trainset_opt, unit=unit.split('_')[0])
        elif len(unit.split('rand')) > 1:
            Algorithm_RAND(trainset_opt=trainset_opt, n=n, num_of_iter=num_of_iter, unit=unit.split('_')[0])
        elif len(unit.split('oracle')) > 1:
            Algorithm_ORACLE(trainset_opt=trainset_opt, n=n, num_of_iter=num_of_iter, unit=unit.split('_')[0])
        else:
            raise NotImplementedError

