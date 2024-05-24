import copy
import random

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
import torch.optim as optim
from torch.distributions import Categorical
import utils

class Buffer:
    def __init__(self, max_len):
         self.buffer = deque(maxlen=max_len)

    def append(self, state, action, reward, next_state, dones, mask):
        transition = (state, action, reward, next_state, dones, mask)

        self.buffer.append(transition)

    def sample(self, batch_size, device):
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        batch_state = [t[0] for t in batch]
        batch_action = [t[1] for t in batch]
        batch_reward = [t[2] for t in batch]
        batch_next_state = [t[3] for t in batch]
        batch_dones = [t[4] for t in batch]
        batch_mask = [t[5] for t in batch]

        batch_state = torch.cat(batch_state, dim=0)
        batch_action = torch.cat(batch_action, dim=0)
        batch_reward = torch.cat(batch_reward, dim=0)
        batch_next_state = torch.cat(batch_next_state, dim=0)
        batch_dones = torch.cat(batch_dones, dim=0)
        batch_mask = torch.cat(batch_mask, dim=0)

        return batch_state, batch_action, batch_reward, batch_next_state, batch_dones, batch_mask

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]


class Actor(nn.Module):
    # state(embedding_dim ) -> action embedding -> action prob
    def __init__(self, state_dim, action_dim, embedding_table):
        super(Actor, self).__init__()
        self.encoding_dim = state_dim
        self.map = nn.Linear(self.encoding_dim + 1, self.encoding_dim)
        self.end_map = nn.Linear(self.encoding_dim + 1, 2)
        # self.map = nn.Sequential(
        #     nn.Linear(self.encoding_dim, self.encoding_dim - 1))  #
        # self.end_map = nn.Sequential(
        #     nn.Linear(self.encoding_dim, 2) )
        self.embedding_table = embedding_table[ :-1].detach() #除去最后一个表示终止的token
        self.temperature = 1
        #self.initialize_mlp()

    def forward(self, state, is_training):
        # TODO 改with torch.no_grad()
        action = self.choose_action(state, is_training)
        return action

    def choose_action(self, s, is_training):
        embedding = self.map(s)
        end = self.end_map(s)
        end_probs = F.softmax(end / self.temperature, dim=-1)
        action_probs = F.softmax(torch.matmul(embedding, self.embedding_table.T) / self.temperature, dim=-1)
        probs = torch.concat([action_probs * end_probs[:, 0].unsqueeze(dim=-1), end_probs[:, 1].unsqueeze(dim=-1)],
                             dim=-1)  # 最后一维为停止的概率
        if is_training:
            sampler = Categorical(probs)
            action = sampler.sample()
        else:
            _,action = torch.max(probs, 1, keepdim=False)

        return action

    def get_log_probs(self, s, action):
        embedding = self.map(s)
        end = self.end_map(s)
        end_probs = F.softmax(end / self.temperature, dim=-1)
        action_probs = F.softmax(torch.matmul(embedding, self.embedding_table.T) / self.temperature, dim=-1)
        probs = torch.concat([action_probs * end_probs[:, 0].unsqueeze(dim=-1) , end_probs[:, 1].unsqueeze(dim=-1)], dim = -1)  # 最后一维为停止的概率
        sampler = Categorical(probs)
        log_probs = sampler.log_prob(action)
        return log_probs

    def action_probs(self, s):
        embedding = self.map(s)
        end = self.end_map(s)
        end_probs = F.softmax(end / self.temperature, dim=-1)
        action_probs = F.softmax(torch.matmul(embedding, self.embedding_table.T) / self.temperature, dim=-1)
        probs = torch.concat([action_probs * end_probs[:, 0].unsqueeze(dim=-1), end_probs[:, 1].unsqueeze(dim=-1)],
                             dim=-1)  # 最后一维为停止的概率

        return probs

    def sample_probs(self, s, sample):
        embedding = self.map(s)
        end = self.end_map(s)
        end_probs = F.softmax(end / self.temperature, dim=-1)
        action_probs = F.softmax(torch.matmul(embedding, self.embedding_table.T) / self.temperature, dim=-1)
        probs = torch.concat([action_probs * end_probs[:, 0].unsqueeze(dim=-1), end_probs[:, 1].unsqueeze(dim=-1)],
                             dim=-1)
        probs = F.softmax(torch.gather(probs,dim=-1,index=sample) / self.temperature)

        return probs

    def state_probs(self, s):
        embedding = self.map(s)
        end = self.end_map(s)
        end_probs = F.softmax(end / self.temperature, dim=-1)
        action_probs = F.softmax(torch.matmul(embedding, self.embedding_table.T) / self.temperature, dim=-1)
        probs = torch.concat([action_probs * end_probs[:, 0].unsqueeze(dim=-1), end_probs[:, 1].unsqueeze(dim=-1)],
                             dim=-1)

        return probs

    def state_probs2(self, s, mask, training = False):
        embedding = self.map(s)

        if training == True:
            mask_porbs = F.softmax(torch.matmul(embedding, self.embedding_table.T) * mask / self.temperature)
            probs = torch.concat([mask_porbs , torch.zeros([len(mask_porbs),1]).to(mask_porbs.device)],
                             dim=-1)
        else:
            end = self.end_map(s)
            end_probs = F.softmax(end / self.temperature, dim=-1)
            mask_porbs = F.softmax(torch.matmul(embedding, self.embedding_table.T) * mask / self.temperature)
            probs = torch.concat([mask_porbs * end_probs[:, 0].unsqueeze(dim=-1), end_probs[:, 1].unsqueeze(dim=-1)],
                                 dim=-1)
        return probs

    def get_log_probs2(self, s, action, mask):
        sample_probs = self.state_probs(s)
        masked_probs = sample_probs * mask
        masked_probs = masked_probs / masked_probs.sum(dim=-1).unsqueeze(dim=1)

        sampler = Categorical(masked_probs)
        log_probs = sampler.log_prob(action)
        return log_probs

    def get_log_probs3(self, s, action, mask):
        masked_probs = self.state_probs2(s, mask)
        sampler = Categorical(masked_probs)
        log_probs = sampler.log_prob(action)
        return log_probs

    def no_repeat_action(self, state, action_list):
        embedding = self.map(state)
        end = self.end_map(state)
        end_probs = F.softmax(end / self.temperature, dim=-1)
        action_probs = F.softmax(torch.matmul(embedding, self.embedding_table.T) / self.temperature, dim=-1)
        probs = torch.concat([action_probs * end_probs[:, 0].unsqueeze(dim=-1), end_probs[:, 1].unsqueeze(dim=-1)],
                             dim=-1)
        if action_list != None:
            one_hot_action = F.one_hot(action_list, num_classes=probs.shape[1]).sum(dim=1)
            action_mask = torch.abs(1 - one_hot_action) # 取反得到mask
            probs = action_mask * probs

        _, action = torch.max(probs, 1, keepdim=False)

        return action
    def state_mask_probs(self, state, mask):
        embedding = self.map(state)
        end = self.end_map(state)
        end_probs = F.softmax(end / self.temperature, dim=-1)
        action_probs = F.softmax(torch.matmul(embedding, self.embedding_table.T) / self.temperature, dim=-1)
        probs = torch.concat([action_probs * end_probs[:, 0].unsqueeze(dim=-1), end_probs[:, 1].unsqueeze(dim=-1)],
                             dim=-1)
        masked_probs = probs * mask
        masked_probs = masked_probs / masked_probs.sum(dim=-1).unsqueeze(dim=1)

        return masked_probs


class Critic(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + 1, 1)

    def forward(self, state):
        score = self.fc1(state)
        return score

# class content_mapping(torch.nn.Module):
#     def __init__(self, content_dim, item_embedding_dim):
#         super(content_mapping, self).__init__()
#         self.fc1 = torch.nn.Linear(content_dim, item_embedding_dim)
#
#     def forward(self, content):
#         score = self.fc1(content)
#         return score
class content_mapping(torch.nn.Module):
    def __init__(self, content_dim, hidden_dim, item_embedding_dim):
        super(content_mapping, self).__init__()
        self.fc1 = torch.nn.Linear(content_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, item_embedding_dim)

    def forward(self, content):
        score = F.relu(self.fc1(content))
        return self.fc2(score)

class USIM(torch.nn.Module):
    """
    不适用token表示终结状态，而是actor单独判断是否结束
    """
    def __init__(self, warm_model, args):
        super(USIM, self).__init__()
        self.warm_model = warm_model.RLmodel()
        self.n_user = len(self.warm_model.user_embedding.weight) # 包含终止动作token
        self.args = args
        self.cold_item_ids = None
        self.warm_item_ids = None
        self.embeddingtable = self.warm_model.user_embedding.weight.to(args.device).detach()

        self.actor = Actor(args.factor_num, self.n_user, self.embeddingtable)

        self.target_actor = Actor(args.factor_num, self.n_user, self.embeddingtable)

        self.critic = Critic(args.factor_num, int(args.factor_num/2))
        self.target_critic = Critic(args.factor_num, int(args.factor_num/2))

        self.CEloss = torch.nn.CrossEntropyLoss()

        #self.content_mapper = content_mapping(args.content_dim , args.factor_num)
        self.content_mapper = content_mapping(args.content_dim, 150,args.factor_num)

        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                               lr=0.0005, weight_decay=1e-6)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=0.001, weight_decay=1e-6)
        self.buffer = Buffer(max_len = 1024)

        #self.one_hot_matirx = torch.eye(self.n_user).to(self.args.device)
        #self.one_hot_matirx = np.eye(self.n_user)

        self.transition_rate = args.transition_rate  #CiteULike最好结果 transition_rate:0.07 max_time:7 ml-1m 0.05 15

        self.max_time = args.max_time

        self.discount = 0.99

        self.lmbda = 0.95

        self.k = self.args.k#固定

        self.k2 = self.args.k

        self.negative_k = args.k #固定

        self.weight = args.weight

        self.reward_cost = args.reward_cost #固定初始0.05

        self.noise_rate = 0#没用

        self.noise_rate2 = 0

        self.hard_update()

    def hard_update(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

    def soft_update(self, scale, type):
        if type == 'critic':
            for target_param, param in zip(self.target_critic.parameters(),
                                           self.critic.parameters()):
                target_param.data.copy_(
                    param.data * scale + target_param.data * (1 - scale))
        if type == 'actor':
            for target_param, param in zip(self.target_actor.parameters(),
                                           self.actor.parameters()):
                target_param.data.copy_(target_param.data)
        if type == 'all':
            for target_param, param in zip(self.target_critic.parameters(),
                                           self.critic.parameters()):
                target_param.data.copy_(
                    param.data * scale + target_param.data * (1 - scale))
            for target_param, param in zip(self.target_actor.parameters(),
                                           self.actor.parameters()):
                target_param.data.copy_(target_param.data)

    def optimize(self, device):
        # update critic
        batch_state, batch_action, batch_reward, batch_next_state, batch_dones, batch_mask = self.buffer.sample(1024 * 20,device)
        batch_dones = batch_dones.float().unsqueeze(1)

        with (torch.no_grad()):
            target_probs = self.target_actor.get_log_probs2(batch_state, batch_action, batch_mask)
            #target_probs = self.target_actor.get_log_probs(batch_state, batch_action)
            target_Q = batch_reward.unsqueeze(1) + self.discount * torch.mul(self.target_critic(batch_next_state), (1 - batch_dones)) #0.99 is the discount
            advantage = target_Q - self.target_critic(batch_state)

        total_actor_loss = 0
        total_critic_loss = 0
        for _ in range(5):

            Q = self.critic(batch_state)
            #advantage = (target_Q - Q).detach()
            log_probs = self.actor.get_log_probs2(batch_state, batch_action, batch_mask)
            #log_probs = self.actor.get_log_probs(batch_state, batch_action)
            ratio = torch.exp(log_probs - target_probs)
            surr1 = ratio * advantage.squeeze()
            surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantage.squeeze() # 截断
            #prob_distri = self.actor.state_mask_probs(batch_state, batch_mask) + 1e-8
            #log = torch.log(prob_distri)
            # if torch.isnan(log).any() == True:
            #     print(log)
            #entropy = - torch.mean(torch.sum(prob_distri * log, dim=1))
            actor_loss = torch.mean(-torch.min(surr1, surr2)) #+ 0.005 * entropy #PPO loss

            # for name, param in self.critic.named_parameters():
            #     param.register_hook(self.print_grad)
            # agu_batch_state1 = torch.rand_like(batch_state).to(self.args.device) * self.noise_rate2 + batch_state
            # agu_batch_state2 = torch.rand_like(batch_state).to(self.args.device) * self.noise_rate2 + batch_state
            # agu_batch_state1 = torch.randn(batch_state.size()).to(self.args.device) * self.noise_rate + batch_state
            # agu_batch_state2 = torch.randn(batch_state.size()).to(self.args.device) * self.noise_rate + batch_state

            # avg_agu_Q = (self.critic(agu_batch_state1) + self.critic(agu_batch_state2)) / 2
            #indices = torch.randperm(Q.shape[0])
            #random_Q = Q[indices]
            #random_Q = self.critic(batch_next_state)
            # contrastive_loss = F.sigmoid(torch.pow(random_Q - avg_agu_Q, 2) - torch.pow(Q - avg_agu_Q, 2))
            # contrastive_loss = - torch.log(contrastive_loss)
            #contrastive_loss = torch.mean(torch.log(F.sigmoid(torch.pow(Q - avg_agu_Q, 2))))
            #contrastive_loss = torch.mean(F.mse_loss(Q, avg_agu_Q))
            critic_loss = torch.mean(F.mse_loss(Q, target_Q.detach())) #+ contrastive_loss.mean()

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            total_actor_loss += actor_loss
            total_critic_loss += critic_loss

        self.hard_update()
        return total_actor_loss, total_critic_loss

    def update_buffer(self, interaction, epoch):
        item = interaction['item']
        target_user = interaction['user']
        item_content = interaction['item_content']

        with torch.no_grad():
            item_embedding = self.warm_model.get_item_embedding(item)
            #state = self.content_mapper(item_content).detach()
            #target_user = self.enlarge_user(target_user, state, item_embedding)

        for i in range(3): #采样3轮
            action_list = None
            dones = torch.zeros(len(item), dtype=torch.bool).to(self.args.device)
            state = self.content_mapper(item_content).detach()
            state_time = torch.zeros(len(item), dtype=torch.float32).to(self.args.device) + self.max_time
            state = state + torch.randn_like(state).to(self.args.device) * self.noise_rate

            for i in range(self.max_time): #每个item最多生成10个user
                with torch.no_grad():
                    action, mask = self.sample_from_negative_distance6(state, item_embedding, i, state_time, action_list)
                    if action_list is None:
                        action_list = action.unsqueeze(dim=1)
                    else:
                        action_list = torch.concat((action_list, action.unsqueeze(dim=1)), dim=1)
                    #dones = self.check_dones(action, dones).detach()
                    reward = self.get_reward(target_user, item_embedding, state,
                                             action, dones).detach()  # batch_size * embedding_dim
                    next_state = self.state_transition(state, action, dones)
                    dones = self.check_dones(action, dones).detach()

                    index_state = torch.concat((state, state_time.unsqueeze(dim=1)), dim=1)
                    state_time = state_time - 1
                    index_next_state = torch.concat((next_state, state_time.unsqueeze(dim=1)), dim=1)
                    self.buffer.append(index_state, action, reward, index_next_state, dones, mask.detach())
                    state = next_state
                    if dones.all() == True:
                        break

        radom_index = torch.randint(low=0,high=self.max_time, size=((len(item_embedding),))).to(self.args.device)
        termination_state = torch.concat((item_embedding, radom_index.unsqueeze(dim=1)), dim=1)
        termination_critic_loss = torch.mean(F.mse_loss(self.critic(termination_state), torch.zeros(item_embedding.shape).to(self.args.device)))
        self.critic_optimizer.zero_grad()
        termination_critic_loss.backward()
        self.critic_optimizer.step()

        # termination_action = (torch.zeros(len(dones),dtype=torch.float32) + self.n_user - 1).to(self.args.device)
        # probs = self.actor.action_probs(termination_state)
        # termination_actor_loss = 0.001*self.CEloss(probs, termination_action.long())
        # self.actor_optimizer.zero_grad()
        # termination_actor_loss.backward()
        # self.actor_optimizer.step()
        #TODO 完善评价指标的计算
        result = 1
        #self.content_mapping_optimizer.zero_grad()
        #content_mapping_loss.backward()
        #self.content_mapping_optimizer.step()
        self.hard_update()

        return result


    def get_reward(self, target_user, item_embedding, state, action, dones):
        #check if predicted action in the action list
        '''
        expanded_tensor = action.expand(-1, action_list.size(1))
        comparison = (action_list == expanded_tensor)
        if_in_action_list = (~comparison.any(dim=1, keepdim=True)).float()

        # check if predicted action in the target_user
        expanded_tensor = action.expand(-1, target_user.size(1))
        comparison = (target_user == expanded_tensor)
        if_in_target_user = comparison.any(dim=1, keepdim=True).float()

        weight = torch.mul(item_embedding, if_in_action_list)
        reward = weight + if_in_target_user
        '''
        with torch.no_grad():
            similarity1 = torch.norm((item_embedding - state),p=2, dim=1, keepdim=True)
            user_embedding = self.warm_model.get_user_embedding(action)
            next_state = state + user_embedding * self.transition_rate
            similarity2 = torch.norm((item_embedding - next_state), p=2, dim=1, keepdim=True)

            target_score = torch.matmul(item_embedding, self.embeddingtable.T)
            top_k_score, index = torch.topk(target_score, 10, dim=-1)

            state_score = torch.matmul(state, self.embeddingtable.T)
            top_k_state_score = torch.gather(state_score, -1, index)

            next_state_score = torch.matmul(next_state, self.embeddingtable.T)
            top_k_next_state_score = torch.gather(next_state_score, -1, index)
            differ_score = torch.abs(top_k_score - top_k_state_score) - torch.abs(top_k_score - top_k_next_state_score)

            log_discount = torch.log2(torch.tensor([i + 2 for i in range(10)])).to(self.args.device)
            differ_score = differ_score / log_discount

            reward1 = self.weight * (similarity1 - similarity2).squeeze() + (1 - self.weight) * differ_score.mean(dim = -1)


            #mask = (similarity2 < self.sim_distance).float().squeeze() * 100
            #reward2 = similarityb - similaritya

            #reward1 = (action == self.n_user - 1).float() * (similarity2 < self.sim_distance).float() + reward1
            reward = (reward1 - self.reward_cost) * (1 - dones.float())   # + reward2 + punishment
            #reward = reward

            #scores_next = self.full_sort_predict(next_state)
            #scores = self.full_sort_predict(state)
            #reward2 = utils.hit_score(target_user, scores_next, 10, sum=False) - utils.hit_score(target_user, scores, 10, sum=False)
            #reward3 = utils.ndcg_score(target_user, scores_next, 10, sum=False) - utils.ndcg_score(target_user, scores, 10, sum=False)

        return reward

    def infer(self, content):
        action_list = None
        dones = torch.zeros(len(content),dtype=torch.bool).to(self.args.device)
        state = self.content_mapper(content)
        state_time = torch.zeros(len(content), dtype=torch.float32).to(self.args.device) + self.max_time
        length= 0
        pre_dones = 0
        for i in range(self.max_time):
            with torch.no_grad():
                index_state = torch.concat((state, state_time.unsqueeze(dim=1)), dim=1)
                action = self.actor(index_state, is_training= False)
                state = self.state_transition(state, action, dones, is_training=False)
                if action_list == None:
                    action_list = action
                else:
                    action_list = torch.cat([action_list,action],dim=-1)
                dones = self.check_dones(action, dones)
            if i == self.max_time - 1:
                now_dones = len(content)
            else:
                now_dones = (dones == True).sum()
            length += (now_dones - pre_dones) * (i + 1)
            pre_dones = now_dones
        avg_length = length / len(content)

        return state,action_list

    def state_transition(self, state, action, dones, is_training = False):
        with torch.no_grad():
            user_embedding = self.warm_model.get_user_embedding(action)
            if is_training:
                user_embedding = user_embedding * (1 - dones.float()).unsqueeze(dim=1)
                next_state = state + user_embedding * self.transition_rate + torch.randn_like(state).to(
                    self.args.device) * self.noise_rate
                # next_state = state - self.transition_rate * 2 * (state - user_embedding)
            else:
                user_embedding = user_embedding * (1 - dones.float()).unsqueeze(dim=1)
                # next_state = state - self.transition_rate * 2 * (state - user_embedding)
                next_state = state + user_embedding * self.transition_rate
        #current_state_rep = state[:, : len(state[0])]
        #current_index = state[:,-1]
        #next_state_rep = current_state_rep + user_embedding * self.transition_rate
        #next_index = current_index - 1
        #next_state = torch.concat((next_state_rep, next_index), dim=0)
        #next_state = state + user_embedding * self.transition_rate

        return next_state

    def check_dones(self, action, dones):
        dones_index = torch.zeros(len(dones)) + self.n_user - 1
        dones_index = dones_index.to(self.args.device)
        if_same = dones_index.eq(action)
        dones = dones|if_same
        return dones

    def buffer_clear(self):
        self.buffer.clear()

    def full_sort_predict(self, item_embedding):
        all_user_embedding = self.warm_model.user_embedding.weight[:-1] #最后一项是终止token
        score = torch.matmul(item_embedding, all_user_embedding.T)
        return score

    def get_pred_MLP(self, path):
        self.content_mapper.load_state_dict(torch.load(path))

    def get_ranked_rating(self, ratings, k):
        top_score, top_item_index = ratings.topk(k, dim=1, largest=True)
        return top_score, top_item_index

    def get_user_rating(self, uids, iids, uemb, iemb):
        out_rating = torch.zeros((len(uids), len(iids)), dtype=torch.float32).to(self.args.device)
        warm_rating = torch.matmul(uemb[uids], iemb[self.warm_item_ids].T)
        out_rating[:, self.warm_item_ids] = warm_rating
        cold_rating = torch.matmul(uemb[uids], iemb[self.cold_item_ids].T)
        out_rating[:, self.cold_item_ids] = cold_rating
        return out_rating

    def get_item_emb(self, item_content, warm_item_ids, cold_item_ids):
        self.warm_item_ids = warm_item_ids
        self.cold_item_ids = cold_item_ids
        item_emb = self.warm_model.item_embedding.weight
        out_emb = copy.deepcopy(item_emb)
        out_emb[cold_item_ids],_ = self.infer(item_content[cold_item_ids])
        #out_emb[cold_item_ids], _ = self.infer_without_repeating(item_content[cold_item_ids])
        #out_emb[cold_item_ids], _ = self.beam_search_generate2(item_content[cold_item_ids])

        return out_emb

    def get_item_emb_step_by_step(self, item_content, warm_item_ids, cold_item_ids):
        self.warm_item_ids = warm_item_ids
        self.cold_item_ids = cold_item_ids
        item_emb = self.warm_model.item_embedding.weight
        cold_item_list,_ = self.infer_step_by_step(item_content[cold_item_ids])
        out_emb_list = []
        for cold_item_embedding in cold_item_list:
            out_emb = copy.deepcopy(item_emb)
            out_emb[cold_item_ids] = cold_item_embedding
            out_emb_list.append(out_emb)
        #out_emb[cold_item_ids], _ = self.infer_without_repeating(item_content[cold_item_ids])
        #out_emb[cold_item_ids], _ = self.beam_search_generate2(item_content[cold_item_ids])

        return out_emb_list

    def get_user_emb(self):
        return self.warm_model.user_embedding.weight[:-1]

    def infer2(self, content, item_embedding):
        action_list = None
        reward_list = None
        distance_list = None
        dones = torch.zeros(len(content),dtype=torch.bool).to(self.args.device)
        state_time = torch.zeros(len(content), dtype=torch.float32).to(self.args.device) + self.max_time
        state = self.content_mapper(content)
        length= 0
        pre_dones = 0
        for i in range(self.max_time):
            with torch.no_grad():
                index_state = torch.concat((state, state_time.unsqueeze(dim=1)),dim=1)
                action = self.actor(index_state, is_training= False)
                next_state = self.state_transition(state, action, dones, is_training = False)
                reward = torch.cosine_similarity(item_embedding, next_state) - torch.cosine_similarity(item_embedding, state)
                distance = torch.norm((item_embedding - next_state),p=2, dim=1, keepdim=True)
                if action_list == None:
                    action_list = action.unsqueeze(dim=1)
                    reward_list = reward.unsqueeze(dim=1)
                    distance_list = distance.unsqueeze(dim=1)
                else:
                    action_list = torch.cat([action_list,action.unsqueeze(dim=1)], dim=-1)
                    reward_list = torch.cat([reward_list,reward.unsqueeze(dim=1)], dim=-1)
                    distance_list = torch.cat([distance_list,distance.unsqueeze(dim=1)], dim=-1)
                dones = self.check_dones(action, dones)
            if i == self.max_time:
                now_dones = len(content)
            else:
                now_dones = (dones == True).sum()
            length += (now_dones - pre_dones) * (i + 1)
            pre_dones = now_dones
            state = next_state
            state_time = state_time - 1
        avg_length = length / len(content)

        return state,action_list,reward_list,distance_list.squeeze()

    def sample_from_negative_distance6(self, state, target_state, i, state_time, action_list):

        distance_vector = target_state - state
        norms = torch.norm(self.warm_model.user_embedding.weight, p=2, dim=-1, keepdim=True)
        norm_mat = self.warm_model.user_embedding.weight / norms
        sim = torch.matmul(distance_vector / torch.norm(distance_vector, p=2, dim=-1,keepdim=True), norm_mat.T).detach()

        state_score = torch.matmul(target_state, self.warm_model.user_embedding.weight[:-1].T).detach()

        _, topk_user = state_score.topk(self.k2, dim=1, largest=True)
        #topk_user = topk_user[:, torch.randperm(topk_user.size(1))]
        #topk_user = topk_user[ : , 1 : -1]

        dones = torch.zeros(len(distance_vector), dtype=torch.int32).to(self.args.device) + self.n_user - 1
        _, positive_sample = sim.topk(self.k, dim=1, largest=True)
        #sim2 = torch.abs(sim)
        #_, neutrality_sample = sim2.topk(self.k, dim=1, largest=False)
        #positive_sample = positive_sample[:, torch.randperm(positive_sample.size(1))]
        #positive_sample = positive_sample[:,1:]

        #从topk个user中和positive_sample中选择80%
        #TODO   将采样改为不重复
        #random_index1 = torch.randint(low=0, high=positive_sample.shape[1], size=(len(positive_sample), int(self.k))).to(self.args.device)
        #random_index2 = torch.randint(low=0, high=topk_user.shape[1],size=(len(topk_user), int(self.k2))).to(self.args.device)
        #positive_sample = torch.gather(positive_sample, dim=-1, index=random_index1)
        #topk_user = torch.gather(topk_user, dim=-1, index=random_index2)

        negative_sample = torch.randint(low=0, high=self.n_user - 1, size=((len(state),self.negative_k))).to(self.args.device)
        # random_num = torch.rand(dones.size()) > 0.5
        # dones = random_num.int().to(self.args.device) * dones
        #random_num = torch.rand(dones.size()) > 0.3
        #sim[:][1] = random_num.int().to(self.args.device) * sim[:][1]

        negative_sample = torch.concat((negative_sample, dones.unsqueeze(dim=-1)), dim=-1)
        # all = torch.concat((topk_user ,positive_sample ,negative_sample, dones.unsqueeze(dim=-1)), dim=-1)
        # one_hot_neg = F.one_hot(negative_sample, num_classes=self.n_user).sum(dim=1)
        one_hot_pos = F.one_hot(positive_sample, num_classes = self.n_user).sum(dim=1) >= 1
        one_hot_neg = F.one_hot(negative_sample, num_classes = self.n_user).sum(dim=1) >= 1
        one_hot_user = F.one_hot(topk_user, num_classes = self.n_user).sum(dim=1) >= 1
        #frq_mask = one_hot_user.sum(dim=0) <20
        #one_hot_pos_mask = F.one_hot(positive_mask, num_classes = self.n_user).sum(dim=1) >= 1
        #one_hot_nue = F.one_hot(neutrality_sample, num_classes = self.n_user).sum(dim=1) >= 1
        # bais_mask = one_hot_user.int().sum(dim=0) <= 15
        # one_hot_user = one_hot_user * bais_mask
        # one_hot_pos = self.one_hot_matirx[positive_sample]
        # one_hot_neg = self.one_hot_matirx[negative_sample]
        # one_hot_user = self.one_hot_matirx[topk_user]
        # one_hot_pos = self.one_hot_matirx[positive_sample].sum(dim=1)
        # one_hot_neg = self.one_hot_matirx[negative_sample].sum(dim=1)
        # one_hot_user = self.one_hot_matirx[topk_user].sum(dim=1)

        # if action_list != None:
        #     mask = torch.logical_or(one_hot_pos, one_hot_user)
        #     #mask = one_hot_pos * one_hot_user
        #     mask = torch.logical_or(mask, one_hot_neg)
        # else:
        #     one_hot_action = F.one_hot(action_list, num_classes = self.n_user).sum(dim=1)
        #     action_mask = torch.abs(1 - one_hot_action)
        #     mask = torch.logical_or(one_hot_pos, one_hot_user)
        #     # mask = one_hot_pos * one_hot_user
        #     mask = torch.logical_or(mask, one_hot_neg)
        #     mask = action_mask * mask

        # mask = torch.logical_or(one_hot_pos, one_hot_user)
        # dropout_mask = (torch.rand(mask.shape) > 0.4).to(self.args.device)
        # mask = mask * dropout_mask

        mask = one_hot_pos * one_hot_user
        #mask = torch.logical_or(one_hot_pos, one_hot_user)
        index = (mask.sum(dim=-1) == 0).nonzero().flatten()
        mask[index] = one_hot_pos[index]

        mask = torch.logical_or(mask, one_hot_neg)
        #mask = mask >= 1

        index_state = torch.concat((state, state_time.unsqueeze(dim=1)), dim=1)
        sample_probs = self.target_actor.state_probs(index_state)

        _, max_index = sample_probs.topk(1, dim=-1)
        random_num = (torch.rand(max_index.size()) > 0.8).to(self.args.device)
        max_index = max_index * random_num
        one_hot_max_index = F.one_hot(max_index, num_classes = self.n_user).sum(dim=1)>= 1
        mask = mask * ~one_hot_max_index

        masked_probs = sample_probs * mask.to(self.args.device)
        masked_probs = masked_probs / masked_probs.sum(dim=-1).unsqueeze(dim=1)

        sampler = Categorical(masked_probs)
        action = sampler.sample()
        #torch.cuda.empty_cache()

        return action, mask

    def infer_step_by_step(self, content):
        action_list = None
        dones = torch.zeros(len(content),dtype=torch.bool).to(self.args.device)
        state = self.content_mapper(content)
        state_time = torch.zeros(len(content), dtype=torch.float32).to(self.args.device) + self.max_time
        length= 0
        pre_dones = 0
        embedding_list = []
        for i in range(self.max_time):
            with torch.no_grad():
                embedding_list.append(state)
                index_state = torch.concat((state, state_time.unsqueeze(dim=1)), dim=1)
                action = self.actor(index_state, is_training= False)
                state = self.state_transition(state, action, dones, is_training=False)
                if action_list == None:
                    action_list = action
                else:
                    action_list = torch.cat([action_list,action],dim=-1)
                dones = self.check_dones(action, dones)
            if i == self.max_time - 1:
                now_dones = len(content)
            else:
                now_dones = (dones == True).sum()
            length += (now_dones - pre_dones) * (i + 1)
            pre_dones = now_dones
        avg_length = length / len(content)

        return embedding_list,action_list








