import copy
import datetime
import os
import pickle
import logging
import random

from metric import ndcg
import numpy
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm

from dataloader.dataloader import RLDataloader
import utils
from warm_model.bprmf import BPRMF
import numpy as np

# class content_mapping(torch.nn.Module):
#     def __init__(self, content_dim, item_embedding_dim):
#         super(content_mapping, self).__init__()
#         self.fc1 = torch.nn.Linear(content_dim, item_embedding_dim)
#         self.dropout = torch.nn.Dropout(p=0.1)
#
#     def forward(self, content):
#         score = self.fc1(content)
#         return score
#
#     def calculate_loss(self, content, item_embedding):
#         content_emb = self.dropout(self.fc1(content))
#         #loss = - torch.cosine_similarity(content_emb, item_embedding)
#         loss = torch.norm((item_embedding - content_emb),p=2, dim=1, keepdim=True)
#         return loss.mean()

class content_mapping(torch.nn.Module):
    def __init__(self, content_dim, hidden_dim, item_embedding_dim):
        super(content_mapping, self).__init__()
        self.fc1 = torch.nn.Linear(content_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, item_embedding_dim)
        self.dropout = torch.nn.Dropout(p=0.001)

    def forward(self, content):
        score = F.relu(self.fc1(content))
        return self.fc2(score)

    def loss_forward(self, content):
        score = F.relu(self.dropout(self.fc1(content)))
        return self.fc2(score)

    def calculate_loss(self, content, item_embedding):

        content_emb = self.loss_forward(content)
        #loss = - torch.cosine_similarity(content_emb, item_embedding)
        loss = torch.norm((item_embedding - content_emb),p=2, dim=1, keepdim=True)
        return loss.mean()

    def get_item_emb(self, item_content, cold_item_ids, item_emb):
        out_emb = copy.deepcopy(item_emb)
        out_emb[cold_item_ids] = self.forward(item_content[cold_item_ids])
        return out_emb



def train_epoch(train_data, epoch_idx, warm_model, model, optimizer, args):
    loss_func = model.calculate_loss
    total_loss = None
    iter_data = tqdm(
        train_data,
        total=len(train_data),
        ncols=100,
        desc=f"Train {epoch_idx:>5}")
    for batch_idx, interaction in enumerate(iter_data):
        interaction.to(args.device)
        content = interaction['item_content']
        item = interaction['item']
        #user = interaction['user']
        optimizer.zero_grad()
        #user_embedding_sum = get_user_embedding_sum(user, warm_model)
        item_embedding =warm_model.get_item_embedding(item)
        losses = loss_func(content, item_embedding)
        if isinstance(losses, tuple):
            loss = sum(losses)
            loss_tuple = tuple(per_loss.item() for per_loss in losses)
            total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
        else:
            loss = losses
            total_loss = losses.item() if total_loss is None else total_loss + losses.item()
        loss.backward()

        optimizer.step()


    return total_loss

def get_user_rating(content_embedding, u_index, i_index):
    user_embedding = warm_model.user_embedding(torch.tensor(u_index).to(args.device))
    item_embedding = copy.deepcopy(warm_model.item_embedding.weight).detach()
    #item_embedding = content_embedding
    item_embedding[para_dict['cold_item']] = content_embedding[para_dict['cold_item']]
    rating = torch.matmul(user_embedding, item_embedding.T)

    return rating


def train_eval(model, warm_model, optimizer, train_data, val_data, logger, args):
    model.to(args.device)

    best_accuracy = 0.0  # 用于保存最佳评估准确率
    best_model_weights = None  # 用于保存最佳模型参数
    early_stopping_counter = 0  # 用于计算连续验证准确率改善次数

    for epoch in range(args.max_epoch):
        running_loss = train_epoch(train_data, epoch, warm_model, model, optimizer, args)
        epoch_loss = running_loss/len(train_data)
        logger.logging(f"Epoch {epoch + 1}/{args.max_epoch}, Training Loss: {epoch_loss:.4f}")

        # 进行评估
        with torch.no_grad():
            content_embedding = model(content_data.float())
            get_user_rating_func = lambda u, v: get_user_rating(content_embedding, u_index=u, i_index=v)
            get_topk = lambda ratings, k: get_top_k(ratings, k)

            # val_res, _ = ndcg.test(get_user_rating_func,
            #                        ts_nei=para_dict['warm_val_user_nb'],
            #                        ts_user=para_dict['warm_val_user'][:args.n_test_user],
            #                        item_array=para_dict['item_array'],
            #                        masked_items=para_dict['cold_item'],
            #                        exclude_pair_cnt=exclude_val_warm,
            #                        )
            #ndcg_list, hit_list = eval_epoch(val_data, epoch, model,args)
            val_res, _ = ndcg.test(get_user_rating_func,
                                    ts_nei=para_dict['cold_test_user_nb'],
                                    ts_user=para_dict['cold_test_user'][:args.n_test_user],
                                    item_array=para_dict['item_array'],
                                    masked_items=para_dict['warm_item'],
                                    exclude_pair_cnt=exclude_test_cold,
                                    )
            # val_res, _ = ndcg.test(get_user_rating_func,
            #                           ts_nei=para_dict['overall_val_user_nb'],
            #                           ts_user=para_dict['overall_val_user'][:args.n_test_user],
            #                           item_array=para_dict['item_array'],
            #                           masked_items=None,
            #                           exclude_pair_cnt=exclude_val_hybrid,
            #                           )


        va_metric_current = val_res['ndcg'][0]
        #va_metric_current = ndcg_list[2]
        # TODO 完善NDCG HIT logger
        logger.logging('Epo%d(%d/%d) loss:%.4f|va_metric:%.4f|Best:%.4f|' %
                       (epoch, early_stopping_counter, args.patience, epoch_loss,
                        va_metric_current, best_accuracy))

        # 保存最佳模型
        if val_res['ndcg'][0] > best_accuracy:
            best_accuracy = val_res['ndcg'][0]
            best_model_weights = copy.deepcopy(model.state_dict())
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # 判断是否提前停止训练
        if early_stopping_counter >= args.patience:
            logger.logging(
                f"Early stopping at epoch {epoch + 1} as validation accuracy didn't improve in {args.patience} epochs.")
            break




    logger.logging("Training complete.")

    # 返回最佳模型参数
    return best_model_weights

def topk_numpy(arr, k, dim):
    idx = numpy.argpartition(-arr,kth=k,axis=dim)
    idx = idx.take(indices=range(k),axis=dim)
    val = numpy.take_along_axis(arr,indices=idx,axis=dim)
    sorted_idx = numpy.argsort(-val,axis=dim)
    idx = numpy.take_along_axis(idx,indices=sorted_idx,axis=dim)
    val = numpy.take_along_axis(val,indices=sorted_idx,axis=dim)
    return val,idx

def get_top_k(ratings, k):
    topk_val, topk_idx = topk_numpy(ratings, k, dim=-1)
    return topk_val, topk_idx



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CiteULike', help='Dataset to use.')#CiteULike,Xing
parser.add_argument('--datadir', type=str, default="data/", help='Director of the dataset.')
parser.add_argument('--device', type=int, default=3)
parser.add_argument('--seed', type=int, default=42, help="Random seed.")
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--reg_rate", type=float, default=1e-3, help="Model regularization rate")
parser.add_argument("--factor_num", type=int, default=200, help="Embedding dimension")
#parser.add_argument('--Ks', nargs='?', default='[20,50,100]', help='Output sizes of every layer.')
parser.add_argument('--test_batch_us', type=int, default=200)
parser.add_argument('--n_test_user', type=int, default=2000)
parser.add_argument('--val_start', type=int, default=1, help="Output beginning point.")
parser.add_argument("--interval", type=int, default=1, help="Output interval.")
parser.add_argument("--patience", type=int, default=10, help="Patience number")
parser.add_argument('--restore', type=str, default="", help="Name of restoring model")
parser.add_argument('--n_jobs', type=int, default=4)
parser.add_argument('--max_epoch', type=int, default=1)
parser.add_argument('--loss', type=str, default='BPR')
parser.add_argument('--Ks', type=str, default='[20]')
parser.add_argument('--backbone_type', type=str, default='MF')


args, _ = parser.parse_known_args()
args.Ks = eval(args.Ks)
ndcg.init(args)

# pprint(vars(args))
#os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
torch.cuda.device_count()
logging.basicConfig(level=logging.INFO)
logger =  utils.Timer(name='main')
data_path = os.path.join(args.datadir, args.dataset)
train_data = RLDataloader(args, 'warm_emb')
val_data = RLDataloader(args,"warm_val")
#test_data = RLDataloader(args,"warm_test")
#test_cold = RLDataloader(args,"cold_item_test")

para_dict = pickle.load(open(os.path.join(data_path, 'convert_dict.pkl'), 'rb'))
if args.dataset == 'ml-1m':
    content_data = torch.load(data_path + f'/{args.dataset}_item_content.pt').to(args.device)
else:
    content_data = np.load(data_path + f'/{args.dataset}_item_content.npy')
    content_data = torch.tensor(content_data).to(args.device)


model = content_mapping(content_data.shape[1], 100, 200).to(args.device)



warm_model = BPRMF(para_dict['user_num'], para_dict['item_num'], args).to(args.device)
warm_model.load_state_dict(torch.load(os.path.join(data_path,'{}_backbone.pt'.format(args.backbone_type))))
warm_model = warm_model.to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
exclude_val_warm = utils.get_exclude_pair_count(para_dict['pos_user_nb'],
                                                    para_dict['warm_val_user'][:args.n_test_user],
                                                    para_dict['warm_val_user_nb'],
                                                    args.test_batch_us)
exclude_test_warm = utils.get_exclude_pair_count(para_dict['pos_user_nb'],
                                                     para_dict['warm_test_user'][:args.n_test_user],
                                                     para_dict['warm_test_user_nb'],
                                                     args.test_batch_us)
exclude_val_cold = utils.get_exclude_pair_count(para_dict['pos_user_nb'],
                                                para_dict['cold_val_user'][:args.n_test_user],
                                                para_dict['cold_val_user_nb'],
                                                args.test_batch_us)
exclude_val_hybrid = utils.get_exclude_pair_count(para_dict['pos_user_nb'],
                                                  para_dict['hybrid_val_user'][:args.n_test_user],
                                                  para_dict['hybrid_val_user_nb'],
                                                  args.test_batch_us)
exclude_test_cold = utils.get_exclude_pair_count(para_dict['pos_user_nb'],
                                                 para_dict['cold_test_user'][:args.n_test_user],
                                                 para_dict['cold_test_user_nb'],
                                                 args.test_batch_us)
exclude_test_hybrid = utils.get_exclude_pair_count(para_dict['pos_user_nb'],
                                                   para_dict['hybrid_test_user'][:args.n_test_user],
                                                   para_dict['hybrid_test_user_nb'],
                                                   args.test_batch_us)


best_weight = train_eval(model, warm_model, optimizer, train_data, val_data, logger, args)
torch.save(best_weight, os.path.join(data_path,'MLP_{}.pt'.format(args.backbone_type)))
model.load_state_dict(best_weight)
#model.load_state_dict(torch.load(os.path.join(data_path,'MLP_lightGCN.pt')))
with torch.no_grad():
    content_embedding = model(content_data.float())
    get_user_rating_func = lambda u, v: get_user_rating(content_embedding, u_index=u, i_index=v)
    get_topk = lambda ratings, k: get_top_k(ratings, k)

# cold recommendation performance

cold_res, _ = ndcg.test(get_user_rating_func,
                        ts_nei=para_dict['cold_test_user_nb'],
                        ts_user=para_dict['cold_test_user'][:args.n_test_user],
                        item_array=para_dict['item_array'],
                        masked_items=para_dict['warm_item'],
                        exclude_pair_cnt=exclude_test_cold,
                        )
# cold_res, _ = ndcg.test(get_user_rating_func,
#                                 ts_nei=para_dict['warm_test_user_nb'],
#                                 ts_user=para_dict['warm_test_user'][:args.n_test_user],
#                                 item_array=para_dict['item_array'],
#                                 masked_items=para_dict['warm_item'],
#                                 exclude_pair_cnt=exclude_test_warm,
#                                 )
logger.logging(
    'Cold-start recommendation result@{}: PRE, REC, NDCG: {:.4f}, {:.4f}, {:.4f}'.format(
        args.Ks[0], cold_res['precision'][0], cold_res['recall'][0], cold_res['ndcg'][0]))

# warm recommendation performance
# warm_res, _ = ndcg.test(get_user_rating_func,
#                         ts_nei=para_dict['cold_test_user_nb'],
#                         ts_user=para_dict['cold_test_user'][:args.n_test_user],
#                         item_array=para_dict['item_array'],
#                         masked_items=para_dict['cold_item'],
#                         exclude_pair_cnt=exclude_test_cold,
#                         )
warm_res, warm_dist = ndcg.test(get_user_rating_func,
                                ts_nei=para_dict['warm_test_user_nb'],
                                ts_user=para_dict['warm_test_user'][:args.n_test_user],
                                item_array=para_dict['item_array'],
                                masked_items=para_dict['cold_item'],
                                exclude_pair_cnt=exclude_test_warm,
                                )
logger.logging("Warm recommendation result@{}: PRE, REC, NDCG: {:.4f}, {:.4f}, {:.4f}".format(
    args.Ks[0], warm_res['precision'][0], warm_res['recall'][0], warm_res['ndcg'][0]))

# hybrid recommendation performance
hybrid_res, _ = ndcg.test(get_user_rating_func,
                          ts_nei=para_dict['hybrid_test_user_nb'],
                          ts_user=para_dict['hybrid_test_user'][:args.n_test_user],
                          item_array=para_dict['item_array'],
                          masked_items=None,
                          exclude_pair_cnt=exclude_test_hybrid,
                          )

logger.logging("hybrid recommendation result@{}: PRE, REC, NDCG: {:.4f}, {:.4f}, {:.4f}".format(
    args.Ks[0], hybrid_res['precision'][0], hybrid_res['recall'][0], hybrid_res['ndcg'][0]))
#test(test_data, 1, model, args)
#test(test_cold, warm_model, model, args)
# with torch.no_grad():
#     gen_item_emb = model.get_item_emb(content_data, para_dict['cold_item'], warm_model.item_embedding.weight)
#     gen_user_emb = warm_model.user_embedding.weight
#     emb_store_path = os.path.join(args.datadir,
#                                   args.dataset,
#                                   "{}_{}.npy".format('mf', "MLP"))
#     np.save(emb_store_path, np.concatenate([gen_user_emb.cpu().numpy(), gen_item_emb.cpu().numpy()], axis=0))
