import copy
import os
import sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import logging
import argparse
from dataloader import RLDataloader
from warm_model.bprmf import BPRMF
import torch
import numpy as np
from metric import ndcg
import utils
from cold_model import USIM


def train_epoch(train_data, epoch_idx, model, args):
    model.train()
    total_actor_loss = 0
    total_critic_loss = 0

    for batch_idx, interaction in enumerate(train_data):
        interaction.to(args.device)
        model.update_buffer(interaction, epoch_idx)
        actor_loss, critic_loss = model.optimize(args.device)
        total_actor_loss += actor_loss
        total_critic_loss += critic_loss

        model.buffer_clear()

    return total_actor_loss, total_critic_loss


def train_RLmodel(model, train_data, logger,  args):
    model.to(args.device)

    best_accuracy = 0.0
    best_model_weights = None
    early_stopping_counter = 0
    user_embedding = model.get_user_emb()
    for epoch in range(args.max_epoch):
        total_actor_loss, total_critic_loss = train_epoch(train_data, epoch, model, args)
        logger.logging(f"Epoch {epoch + 1}/{args.max_epoch}, total_actor_loss:{total_actor_loss/ len(train_data):.4f}, total_critic_loss: {total_critic_loss/ len(train_data):.4f}")

        #进行评估

        model.eval()
        with torch.no_grad():
            item_embedding =  model.get_item_emb(content_data.float(),para_dict['warm_item'], para_dict['cold_item'])
            va_metric, _ = ndcg.test(
                lambda u, i: model.get_user_rating(u, i, user_embedding, item_embedding),
                ts_nei=para_dict['cold_val_user_nb'],
                ts_user=para_dict['cold_val_user'][:args.n_test_user],
                item_array=para_dict['item_array'],
                masked_items=para_dict['warm_item'],
                exclude_pair_cnt=exclude_val_cold,
            )
        va_metric_current =  va_metric['ndcg'][0]
        # TODO 完善NDCG HIT logger
        logger.logging('Epo%d(%d/%d) actor_loss:%.4f|critic_loss:%.4f|va_metric:%.4f|Best:%.4f|' %
                      (epoch, early_stopping_counter, args.patience, total_actor_loss, total_critic_loss,
                       va_metric_current, best_accuracy))

        # 保存最佳模型
        if va_metric['ndcg'][0] > best_accuracy:
            best_accuracy = va_metric['ndcg'][0]
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

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
parser = argparse.ArgumentParser()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
parser.add_argument('--n_jobs', type=int, default=4, help='Multiprocessing number.')
parser.add_argument('--dataset', type=str, default="CiteULike", help='Dataset to use.')#ml-1m,CiteULike,XING
parser.add_argument('--datadir', type=str, default="data/", help='Director of the dataset.')
parser.add_argument('--device', type=int, default=2)
parser.add_argument('--seed', type=int, default=42, help="Random seed.")
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--factor_num", type=int, default=200, help="Embedding dimension")
parser.add_argument('--test_batch_us', type=int, default=200)
parser.add_argument("--interval", type=int, default=1, help="Output interval.")
parser.add_argument("--patience", type=int, default=10, help="Patience number")
parser.add_argument('--restore', type=str, default="", help="Name of restoring model")
parser.add_argument('--max_epoch', type=int, default=1)
parser.add_argument('--n_test_user', type=int, default=2000)
parser.add_argument('--Ks', type=str, default='[20]', help='Top K recommendation')
parser.add_argument('--max_time', type=int, default=7)
parser.add_argument('--transition_rate', type=float, default=0.05)
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--reward_cost', type=float, default=0.2)
parser.add_argument('--weight', type=float, default=0.5)
parser.add_argument('--backbone_type', type=str, default='MF')

logging.basicConfig(level=logging.INFO)
args, _ = parser.parse_known_args()
#! To check
#wandb.init(config=args, reinit=True)
args.Ks = eval(args.Ks)
ndcg.init(args)
 # citeULike 300, ml-1m 200
# pprint(vars(args))
#os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
logger = utils.Timer(name='main')
data_path = os.path.join(args.datadir, args.dataset)
para_dict = pickle.load(open(os.path.join(data_path, 'convert_dict.pkl'), 'rb'))

train_data = RLDataloader(args, 'warm_emb')
warm_test_data = RLDataloader(args, "warm_test")

data_path = os.path.join(args.datadir, args.dataset)


if args.dataset == 'ml-1m':
    content_data = torch.load(data_path + f'/{args.dataset}_item_content.pt').to(args.device)
else:
    content_data = np.load(data_path + f'/{args.dataset}_item_content.npy')
    content_data = torch.tensor(content_data).float().to(args.device)
args.content_dim = content_data.shape[1]



exclude_val_cold = utils.get_exclude_pair_count(para_dict['pos_user_nb'],
                                                para_dict['cold_val_user'][:args.n_test_user],
                                                para_dict['cold_val_user_nb'],
                                                args.test_batch_us)
exclude_val_hybrid = utils.get_exclude_pair_count(para_dict['pos_user_nb'],
                                                  para_dict['hybrid_val_user'][:args.n_test_user],
                                                  para_dict['hybrid_val_user_nb'],
                                                  args.test_batch_us)
exclude_test_warm = utils.get_exclude_pair_count(para_dict['pos_user_nb'],
                                                 para_dict['warm_test_user'][:args.n_test_user],
                                                 para_dict['warm_test_user_nb'],
                                                 args.test_batch_us)
exclude_test_cold = utils.get_exclude_pair_count(para_dict['pos_user_nb'],
                                                 para_dict['cold_test_user'][:args.n_test_user],
                                                 para_dict['cold_test_user_nb'],
                                                 args.test_batch_us)
exclude_test_hybrid = utils.get_exclude_pair_count(para_dict['pos_user_nb'],
                                                   para_dict['hybrid_test_user'][:args.n_test_user],
                                                   para_dict['hybrid_test_user_nb'],
                                                   args.test_batch_us)


warm_model = BPRMF(para_dict['user_num'], para_dict['item_num'], args)
warm_model.load_state_dict(torch.load(os.path.join(data_path, args.backbone_type + '_backbone.pt')))
warm_model.to(args.device)
# with torch.no_grad():
#     gen_user_emb = warm_model.get_user_emb()
#     gen_item_emb = warm_model.get_item_emb(content_data, para_dict['warm_item'], para_dict['cold_item'])
# get_user_rating_func = lambda u, v: warm_model.get_user_rating( u, v)
# get_topk = lambda rat, k: warm_model.get_ranked_rating(rat, k)
#
# ts_res, _ = ndcg.test(get_topk, get_user_rating_func,
#                           ts_nei=para_dict['warm_test_user_nb'],
#                           ts_user=para_dict['warm_test_user'][:args.n_test_user],
#                           item_array=para_dict['item_array'],
#                           masked_items=para_dict['cold_item'],
#                           exclude_pair_cnt=exclude_test_warm,
#                           )
# logger.logging(
#         '[Test] Time Pre Rec nDCG: ' +
#         '{:.4f} {:.4f} {:.4f}'.format(ts_res['precision'][0], ts_res['recall'][0], ts_res['ndcg'][0]))
# warm recommendation performance
model = USIM(warm_model, args)

model.to(args.device)
warm_model.to(args.device)
#model.load_content_table(os.path.join(data_path,'bprmf_DEEPMUSIC.npy'))
if args.dataset != "XING":
    model.get_pred_MLP(os.path.join(data_path,'MLP_{}.pt'.format(args.backbone_type)))
#testMLP(val_data, cold_test_data, warm_model, model.content_mapper, args)
best_model_weight = train_RLmodel(model, train_data, logger, args)
torch.save(best_model_weight, os.path.join(data_path,'{}_cold_model.pt'.format(args.backbone_type)))
model.load_state_dict(best_model_weight)
#model.load_state_dict(torch.load(os.path.join(data_path,f'{args.backbone_type}_cold_model.pt')))

with torch.no_grad():
    gen_user_emb = model.get_user_emb()
    gen_item_emb = model.get_item_emb(content_data, para_dict['warm_item'], para_dict['cold_item'])

    # cold recommendation performance
    cold_res, _ = ndcg.test(
                            lambda u, i: model.get_user_rating(u, i, gen_user_emb, gen_item_emb),
                            ts_nei=para_dict['cold_test_user_nb'],
                            ts_user=para_dict['cold_test_user'][:args.n_test_user],
                            item_array=para_dict['item_array'],
                            masked_items=para_dict['warm_item'],
                            exclude_pair_cnt=exclude_test_cold,
                            )
    # cold_res, _ = ndcg.test(
    #     lambda u, i: model.get_user_rating(u, i, gen_user_emb, gen_item_emb),
    #     ts_nei=para_dict['cold_test_user_nb'],
    #     ts_user=para_dict['cold_test_user'][:args.n_test_user],
    #     item_array=para_dict['item_array'],
    #     masked_items=para_dict['warm_item'],
    #     exclude_pair_cnt=exclude_test_cold,
    # )
    logger.logging(
        'Cold-start recommendation result@{}: PRE, REC, NDCG: {:.4f}, {:.4f}, {:.4f}'.format(
            args.Ks[0], cold_res['precision'][0], cold_res['recall'][0], cold_res['ndcg'][0]))
    #wandb.log({f'cold_pre': cold_res['precision'][0], f'cold_rec':cold_res['recall'][0], f'cold_NDCG':cold_res['ndcg'][0]})

    # warm recommendation performance
    warm_res, warm_dist = ndcg.test(
                                    lambda u, i: model.get_user_rating(u, i, gen_user_emb, gen_item_emb),
                                    ts_nei=para_dict['warm_test_user_nb'],
                                    ts_user=para_dict['warm_test_user'][:args.n_test_user],
                                    item_array=para_dict['item_array'],
                                    masked_items=para_dict['cold_item'],
                                    exclude_pair_cnt=exclude_test_warm,
                                    )
    logger.logging("Warm recommendation result@{}: PRE, REC, NDCG: {:.4f}, {:.4f}, {:.4f}".format(
        args.Ks[0], warm_res['precision'][0], warm_res['recall'][0], warm_res['ndcg'][0]))
    # wandb.log(
    #     {f'warm_pre': warm_res['precision'][0], f'warm_rec': warm_res['recall'][0], f'warm_NDCG': warm_res['ndcg'][0]})

    # hybrid recommendation performance
    hybrid_res, _ = ndcg.test(
                              lambda u, i: model.get_user_rating(u, i, gen_user_emb, gen_item_emb),
                              ts_nei=para_dict['hybrid_test_user_nb'],
                              ts_user=para_dict['hybrid_test_user'][:args.n_test_user],
                              item_array=para_dict['item_array'],
                              masked_items=None,
                              exclude_pair_cnt=exclude_test_hybrid,
                              )

    logger.logging("hybrid recommendation result@{}: PRE, REC, NDCG: {:.4f}, {:.4f}, {:.4f}".format(
        args.Ks[0], hybrid_res['precision'][0], hybrid_res['recall'][0], hybrid_res['ndcg'][0]))
    # wandb.log(
    #     {f'hybrid_pre': hybrid_res['precision'][0], f'hybrid_rec': hybrid_res['recall'][0], f'hybrid_NDCG': hybrid_res['ndcg'][0]})
    #test_RLmodel(model, warm_test_data, train_data, logger, args)
    emb_store_path = os.path.join(args.datadir,
                                  args.dataset,
                                  "{}_{}.npy".format('mf',"RL4Rec"))
    np.save(emb_store_path, np.concatenate([gen_user_emb.cpu().numpy(), gen_item_emb.cpu().numpy()], axis=0))