from copy import copy, deepcopy

import torch
import torch.nn as nn
import utils
import torch.nn.functional as F
import copy
class content_mapping(torch.nn.Module):
    def __init__(self, content_dim, hidden_dim, item_embedding_dim):
        super(content_mapping, self).__init__()
        self.fc1 = torch.nn.Linear(content_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, item_embedding_dim)

    def forward(self, content):
        score = F.relu(self.fc1(content))
        return self.fc2(score)
class BPRMF(nn.Module):
    def __init__(self, n_user, n_item, args):
        super(BPRMF, self).__init__()
        self.cold_item_ids = None
        self.warm_item_ids = None
        self.n_user = n_user
        self.n_item = n_item
        self.args = args
        self.embedding_size = args.factor_num
        self.user_embedding = nn.Embedding(self.n_user, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_item, self.embedding_size)
        #self.content_maper = content_mapping(300, 150, 200)
        #self.user_weight = deepcopy(self.user_embedding.weight)
        #self.item_weight = deepcopy(self.item_embedding.weight)
        self.apply(utils.xavier_normal_initialization)


    def get_user_embedding(self, user):
        r""" Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        r""" Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding(item)

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e

    def calculate_loss(self, interaction):
        user = interaction[:, 0]
        pos_item = interaction[:, 1]
        neg_item = interaction[:, 2]

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        loss = -torch.log(1e-10 + torch.sigmoid(pos_item_score - neg_item_score)).mean()

        return loss

    def predict(self, interaction):
        user = interaction[:, 0]
        item = interaction[:, 1]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[:, 0]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score

    def RLmodel(self):
        model = BPRMF(self.n_user + 1, self.n_item + 1, self.args)
        user_embedding = torch.cat([self.user_embedding.weight,torch.zeros(self.embedding_size).unsqueeze(0).to(self.args.device) ],dim=0)

        model.user_embedding.weight = nn.Parameter(user_embedding)
        item_embedding = self.item_embedding.weight
        model.item_embedding.weight = nn.Parameter(item_embedding)

        return model

    def full_sort_predict_item(self, item):
        item_e = self.get_item_embedding(item)
        all_item_u = self.user_embedding.weight
        score = torch.matmul(item_e, all_item_u.transpose(0, 1))
        return score

    def full_sort_predict_item_content(self, item_embedding):
        all_item_u = self.user_embedding.weight
        score = torch.matmul(item_embedding, all_item_u.transpose(0, 1))
        return score

    def pertrain_emb(self, user_emb, item_emb):
        self.user_embedding.weight = nn.Parameter(user_emb.to(self.args.device))
        self.item_embedding.weight = nn.Parameter(item_emb.to(self.args.device))
        #self.content_maper.load_state_dict(content_maper_weight)

    def get_user_emb(self):
        return self.user_embedding.weight

    def infer(self, item_content):
        return self.content_maper(item_content)

    def get_item_emb(self, item_content, warm_item_ids, cold_item_ids):
        self.warm_item_ids = warm_item_ids
        self.cold_item_ids = cold_item_ids
        item_emb = self.item_embedding.weight
        out_emb = copy.deepcopy(item_emb)
        out_emb[cold_item_ids] = self.infer(item_content[cold_item_ids])

        return out_emb

    def get_user_rating(self, uids, iids):
        i = self.item_embedding(torch.tensor(iids).to(self.args.device))
        return torch.sigmoid(torch.matmul(self.user_embedding(torch.tensor(uids).to(self.args.device)), i.T))
    def get_ranked_rating(self, ratings, k):
        top_score, top_item_index = ratings.topk(k, dim=1, largest=True)
        return top_score, top_item_index












