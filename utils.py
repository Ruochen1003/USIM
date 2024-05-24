import time
import numpy as np
import torch
import random
import os
import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_


def df_get_neighbors(input_df, obj, max_num):
    """
    Get users' neighboring items.
    return:
        nei_array - [max_num, neighbor array], use 0 to pad users which have no neighbors.
    """
    group = tuple(input_df.groupby(obj))
    keys, values = zip(*group)  # key: obj scalar, values: neighbor array

    keys = np.array(keys, dtype=np.int64)
    opp_obj = 'item' if obj == 'user' else 'user'
    values = list(map(lambda x: x[opp_obj].values, values))
    values.append(0)
    values = np.array(values, dtype=object)

    nei_array = np.zeros((max_num,), dtype=object)
    nei_array[keys] = values[:-1]
    return nei_array




def bpr_neg_samp(uni_users, n_users, support_dict, item_array):
    """
    :parameter:
        uni_users - unique users in training data
        dict - {uid: array[items]}
        n_users - sample n users
        neg_num - n of sample pairs for a user.
        item_array - sample item in this array.

    :return:
        ret_array - [uid pos_iid neg_iid] * n_records
    """
    pos_items = []
    users = np.random.choice(uni_users, size=n_users, replace=True)
    for user in users:
        # pos sampling
        pos_candidates = support_dict[user]
        # if not hasattr(pos_candidates, 'shape'):
        #     continue
        pos_item = random.choice(pos_candidates)
        pos_items.append(pos_item)

    pos_items = np.array(pos_items, dtype=np.int32).flatten()
    neg_items = np.random.choice(item_array, len(users), replace=True)
    ret = np.stack([users, pos_items, neg_items], axis=1)
    return ret


def negative_sampling(pos_user_array, pos_item_array, neg, warm_item):
    """
    :parameter:
        pos_user_array: users in train interactions
        pos_item_array: items in train interactions
        neg: num of negative samples
        warm_item: train item set

    :return:
        user: concat pos users and neg ones
        item: concat pos item and neg ones
        target: scores of both pos interactions and neg ones
    """
    user_pos = pos_user_array.reshape((-1))
    if neg >= 1:
        user_neg = np.tile(pos_user_array, int(neg)).reshape((-1))
    else:
        user_neg = np.random.choice(pos_user_array, size=(int(neg * len(user_pos))), replace=True)
    user_array = np.concatenate([user_pos, user_neg], axis=0)
    item_pos = pos_item_array.reshape((-1))
    item_neg = np.random.choice(warm_item, size=user_neg.shape[0], replace=True).reshape((-1))
    item_array = np.concatenate([item_pos, item_neg], axis=0)
    target_pos = np.ones_like(item_pos)
    target_neg = np.zeros_like(item_neg)
    target_array = np.concatenate([target_pos, target_neg], axis=0)
    random_idx = np.random.permutation(user_array.shape[0])  # 生成一个打乱的 range 序列作为下标
    return user_array[random_idx], item_array[random_idx], target_array[random_idx]


def get_exclude_pair(pos_user_nb, u_pair, ts_nei):
    """Find the items in the complete dataset but not in the test set for a user"""
    pos_item = np.array(list(set(pos_user_nb[u_pair[0]]) - set(ts_nei[u_pair[0]])),
                        dtype=np.int64)
    pos_user = np.array([u_pair[1]] * len(pos_item), dtype=np.int64)
    return np.stack([pos_user, pos_item], axis=1)


def get_exclude_pair_count(pos_user_nb, ts_user, ts_nei, batch):
    exclude_pair_list = []
    exclude_count = [0]
    for i, beg in enumerate(range(0, len(ts_user), batch)):
        end = min(beg + batch, len(ts_user))
        batch_user = ts_user[beg:end]
        batch_range = list(range(end - beg))
        batch_u_pair = tuple(zip(batch_user.tolist(), batch_range))  # (org_id, map_id)

        specialize_get_exclude_pair = lambda x: get_exclude_pair(pos_user_nb, x, ts_nei)
        exclude_pair = list(map(specialize_get_exclude_pair, batch_u_pair))
        exclude_pair = np.concatenate(exclude_pair, axis=0)

        exclude_pair_list.append(exclude_pair)
        exclude_count.append(exclude_count[i] + len(exclude_pair))

    exclude_pair_list = np.concatenate(exclude_pair_list, axis=0)
    return [exclude_pair_list, exclude_count]

def xavier_normal_initialization(module):
    r""" using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.

    .. _`xavier_normal_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_

    Examples:
        >>> self.apply(xavier_normal_initialization)
    """
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


def ndcg_score(y_true, y_pred, k, sum=True):
    _, indices = torch.topk(y_pred, k, sorted=True)
    interactions = torch.zeros_like(y_pred)
    for i in range(len(interactions)):
        interactions[i][y_true[i]] = 1

    ranked_interactions = interactions.gather(1, indices)
    discounts = torch.log2(torch.arange(2, k + 2).float()).to(y_pred.device)
    dcg = ranked_interactions / discounts
    dcg = torch.sum(dcg, dim=-1)

    perfect_ranking = interactions.sort(descending=True)[0][:, :k]
    perfect_discounts = torch.log2(torch.arange(2, k + 2).float()).to(y_pred.device)
    idcg = perfect_ranking / perfect_discounts
    idcg = torch.sum(idcg, dim=-1)
    if sum == True:
        ndcg = (dcg / idcg).sum().item()
    else:
        ndcg = dcg / idcg
    return ndcg

def ndcg_score2(y_true, y_pred, k):
    assert len(y_pred) == len(y_true)
    pred_data = y_pred[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(y_true):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(20 + 2) + 1e-9)[2:k + 2], axis=1)
    dcg = pred_data.cpu().numpy() * (1. / np.log2(np.arange(20 + 2) + 1e-9)[2:k + 2])
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def hit_score(y_true, y_pred, k, sum=True):
    _, indices = torch.topk(y_pred, k, sorted=True)
    interactions = torch.zeros_like(y_pred)
    for i in range(len(interactions)):
        interactions[i][y_true[i]] = 1
    hits = torch.sum(interactions.gather(1, indices), dim=1)
    if sum == True:
        hit_rate = (hits / k).sum().item()
    else:
        hit_rate = (hits / k)
    return hit_rate

class Timer(object):
    def __init__(self, name=''):
        self._name = name
        self.begin_time = time.time()
        self.last_time = time.time()
        self.current_time = time.time()
        self.stage_time = 0.0
        self.run_time = 0.0

    def logging(self, message):
        """
        output the time information, including current datetime, time of duration, message

        :parameter:
            message - operation information
        """
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.update()
        message = '' if message is None else message
        print("{} {} {:.0f}s {:.0f}s | {}".format(current_time,
                                                  self._name,
                                                  self.run_time,
                                                  self.stage_time,
                                                  message))

    def update(self):
        self.current_time = time.time()

        self.stage_time = self.current_time - self.last_time
        self.last_time = self.current_time
        self.run_time = self.current_time - self.begin_time
        return self