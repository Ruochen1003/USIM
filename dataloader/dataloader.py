import pickle

import numpy as np
import torch
import os
import pandas as pd
import utils
from dataloader.Interaction import interaction


class BPRDataLoader(object):
    def __init__(self, args, type, para_dict):
        self.data_path = os.path.join("../data/",args.dataset,type)
        self.m_item = 0
        self.cnt = 0
        self.batch_size = args.batch_size
        self.step = self.batch_size
        self.df = pd.read_csv(self.data_path+'.csv', dtype=np.int64).values
        self.para_dict =para_dict
        self.df = utils.bpr_neg_samp(
            uni_users=self.para_dict['warm_user'],
            n_users=len(self.df),
            support_dict=self.para_dict['emb_user_nb'],
            item_array=self.para_dict['warm_item'],
        )
        self.pr = 0
        self.pr_end = len(self.df)

    def _next_batch_data(self):
        cur_data = self.df[self.pr:self.pr + self.step]
        self.pr += self.step
        return torch.tensor(cur_data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.pr >= self.pr_end:
            self.pr = -1
            return torch.tensor(self.df[self.pr - self.step:])
        elif self.pr == -1:
            self.pr = 0
            raise StopIteration()
        return self._next_batch_data()

    def __len__(self):
        return int(len(self.df)/self.batch_size)

    def negative_sampling(self):
        self.df = utils.bpr_neg_samp(
            uni_users=self.para_dict['warm_user'],
            n_users=len(self.df),
            support_dict=self.para_dict['emb_user_nb'],
            item_array=self.para_dict['warm_item'],
        )

class BPREvalDataLoader(object):
    def __init__(self, args, type, para_dict):
        self.data_path = os.path.join("../data/",args.dataset,type)
        self.m_item = 0
        self.cnt = 0
        self.batch_size = args.batch_size
        self.step = self.batch_size
        self.df = pd.read_csv(self.data_path+'.csv', dtype=np.int64).values
        self.df = utils.bpr_neg_samp(
            uni_users=para_dict['warm_user'],
            n_users=len(self.df),
            support_dict=para_dict['emb_user_nb'],
            item_array=para_dict['warm_item'],
        )
        self.pr = 0
        self.pr_end = len(self.df)

    def _next_batch_data(self):
        cur_data = self.df[self.pr:self.pr + self.step]
        self.pr += self.step
        return torch.tensor(cur_data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.pr >= self.pr_end:
            self.pr = -1
            return torch.tensor(self.df[self.pr - self.step:])
        elif self.pr == -1:
            self.pr = 0
            raise StopIteration()
        return self._next_batch_data()

    def __len__(self):
        return int(len(self.df)/self.batch_size)



class RLDataloader(object):
    def __init__(self, args, type):
        self.data_path = os.path.join("data/", args.dataset)
        self.batch_size = args.batch_size
        self.step = self.batch_size
        self.df = pd.read_csv(os.path.join(self.data_path,type) + '.csv', dtype=np.int64).groupby('item')['user'].agg(list).to_dict()
        if args.dataset == 'ml-1m':
            self.all_item_content = torch.load(self.data_path + f'/{args.dataset}_item_content.pt')
        else:
            self.all_item_content = np.load(self.data_path + f'/{args.dataset}_item_content.npy')
        self.interaction = None

        self._data_processing()
        self.pr = 0
        self.pr_end = len(self.interaction['item'])

    def _data_processing(self):
        if torch.is_tensor(self.all_item_content[0]):
            item_content = [self.all_item_content[item_id].unsqueeze(dim=0) for item_id in self.df.keys()]
        else:
            item_content = [self.all_item_content[item_id] for item_id in self.df.keys()]
        self.interaction = {"item":list(self.df.keys()), "user":list(self.df.values()), "item_content":item_content}
        #self.interaction = {"user": list(self.df.keys()), "item": list(self.df.values()), "item_content": item_content}
        self.interaction = interaction(self.interaction)

    def __iter__(self):
        return self

    def __next__(self):
        if self.pr >= self.pr_end:
            self.pr = -1
            return self.interaction[self.pr - self.step:]
        elif self.pr == -1:
            self.pr = 0
            raise StopIteration()
        return self._next_batch_data()

    def _next_batch_data(self):
        cur_data = self.interaction[self.pr:self.pr + self.step]
        self.pr += self.step
        return cur_data

    def __len__(self):
        return int(len(self.interaction)/self.batch_size)


class RLDataloader_Emb(object):
    def __init__(self, args, type):
        self.data_path = os.path.join("../data/", args.dataset)
        self.batch_size = args.batch_size
        self.step = self.batch_size
        self.df = pd.read_csv(os.path.join(self.data_path,type) + '.csv', dtype=np.int64).groupby('item')['user'].agg(list).to_dict()
        self.all_item_embedding = np.load(self.data_path + f'/{args.backbone_type}_{args.other_ini_type}_embedding.npy')
        self.interaction = None

        self._data_processing()
        self.pr = 0
        self.pr_end = len(self.interaction['item'])

    def _data_processing(self):

        item_embedding = [self.all_item_embedding[item_id] for item_id in self.df.keys()]
        self.interaction = {"item":list(self.df.keys()), "user":list(self.df.values()), "item_content":item_embedding}
        #self.interaction = {"user": list(self.df.keys()), "item": list(self.df.values()), "item_content": item_content}
        self.interaction = interaction(self.interaction)

    def __iter__(self):
        return self

    def __next__(self):
        if self.pr >= self.pr_end:
            self.pr = -1
            return self.interaction[self.pr - self.step:]
        elif self.pr == -1:
            self.pr = 0
            raise StopIteration()
        return self._next_batch_data()

    def _next_batch_data(self):
        cur_data = self.interaction[self.pr:self.pr + self.step]
        self.pr += self.step
        return cur_data

    def __len__(self):
        return int(len(self.interaction)/self.batch_size)




