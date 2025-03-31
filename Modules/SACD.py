# -*- encoding: utf-8 -*-
"""
@File    : SACD.py
@Time    : 2023/11/9 17:25
@Author  : junruitian
@Software: PyCharm
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical

from Configuration.util import ReplayBuffer


class Policy_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Policy_Net, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]
        self.P = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, s):
        logits = self.P(s)
        probs = F.softmax(logits, dim=1)
        return probs


def build_net(layer_shape, hid_activation, output_activation):
    '''
        build net with for loop
    '''
    layers = []
    for j in range(len(layer_shape) - 1):
        act = hid_activation if j < len(layer_shape) - 2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
    return nn.Sequential(*layers)


class Double_Q_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Double_Q_Net, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]

        self.Q1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, s):
        q1 = self.Q1(s)
        q2 = self.Q2(s)
        return q1, q2


class SACD_Agent(object):
    def __init__(self, action_dim, state_dim, gamma, lr, hide_shape, alpha, batch_size, adaptive_alpha, dvc):
        super(SACD_Agent, self).__init__()
        self.model_name = 'SACD_Agent'
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = 0.005
        self.state_dim = state_dim
        self.hide_shape = hide_shape
        self.learning_rate = lr
        self.dvc = dvc
        self.replay_buffer = []
        for b in range(0, self.batch_size):
            self.replay_buffer.append(ReplayBuffer(state_dim=self.state_dim, dvc=self.dvc, max_size=int(1e5)))

        self.actor = Policy_Net(self.state_dim, self.action_dim, self.hide_shape).to(self.dvc)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        self.q_critic = Double_Q_Net(self.state_dim, self.action_dim, self.hide_shape).to(self.dvc)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.learning_rate)
        self.q_critic_target = copy.deepcopy(self.q_critic)
        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        self.alpha = alpha
        self.adaptive_alpha = adaptive_alpha
        if self.adaptive_alpha:
            # We use 0.6 because the recommended 0.98 will cause alpha explosion.
            self.target_entropy = 0.6 * (-np.log(1 / action_dim))  # H(discrete)>0
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.dvc)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.H_mean = 0

    def save(self, timestep, EnvName):
        torch.save(self.actor.state_dict(), f"./model/sacd_actor_{timestep}_{EnvName}.pth")
        torch.save(self.q_critic.state_dict(), f"./model/sacd_critic_{timestep}_{EnvName}.pth")

    def load(self, timestep, EnvName):
        self.actor.load_state_dict(torch.load(f"./model/sacd_actor_{timestep}_{EnvName}.pth"))
        self.q_critic.load_state_dict(torch.load(f"./model/sacd_critic_{timestep}_{EnvName}.pth"))

    def select_action(self, state_batch, deterministic):
        for idx in range(0, state_batch.shape[0]):
            state = state_batch[idx]
            with torch.no_grad():
                # print(type(state))
                if isinstance(state, np.ndarray):
                    state = torch.FloatTensor(state[np.newaxis, :]).to(self.dvc)  # from (s_dim,) to (1, s_dim)
                elif isinstance(state, torch.Tensor):
                    state = state[np.newaxis, :]  # from (s_dim,) to (1, s_dim)
                # state = torch.FloatTensor([state]).to(self.dvc)
                # Double to float
                state_float = state.to(torch.float)
                # print(state_float)
                probs = self.actor(state_float)
                # print(probs)
                if deterministic:
                    a = probs.argmax(-1).item()
                else:
                    a = Categorical(probs).sample().item()
                a = probs
                if idx == 0:
                    action_res = a
                else:
                    action_res = torch.cat((action_res, a), dim=0)

        return action_res

    def train(self):
        for idx in range(0, self.batch_size):
            s_, a_, r_, s_next_, dw = self.replay_buffer[idx].sample()
            if idx == 0:
                s, a, r, s_next = s_, a_, r_, s_next_
            else:
                s = torch.cat((s, s_), dim=0)
                a = torch.cat((a, a_), dim=0)
                r = torch.cat((r, r_), dim=0)
                s_next = torch.cat((s_next, s_next_), dim=0)
            # print(s.shape, a.shape)
        # ------------------------------------------ Train Critic ----------------------------------------#
        '''Compute the target soft Q value'''
        with torch.no_grad():
            next_probs = self.actor(s_next)  # [b,a_dim]
            next_log_probs = torch.log(next_probs + 1e-8)  # [b,a_dim]
            next_q1_all, next_q2_all = self.q_critic_target(s_next)  # [b,a_dim]
            min_next_q_all = torch.min(next_q1_all, next_q2_all)
            v_next = torch.sum(next_probs * (min_next_q_all - self.alpha * next_log_probs), dim=1,
                               keepdim=True)  # [b,1]
            target_Q = r + (~dw) * self.gamma * v_next

        '''Update soft Q net'''
        q1_all, q2_all = self.q_critic(s)  # [b,a_dim]
        # print(q1_all.shape, q2_all.shape) # [24, 300]

        # test adding
        # 两个张量 a 和 q，形状分别为 (batch, 300)，并且想要通过 gather 操作从 q 中选择 a 中对应的元素，将 q 降维到形状 (batch, 1)，
        seed_value = 42
        torch.manual_seed(seed_value)
        #indices = torch.randint(a.shape[1], (a.shape[0], 1)).to(self.dvc)

        
        q1, q2 = q1_all.gather(1, a), q2_all.gather(1, a)  # [b,1]

        # 基于a的维度进行修改 原SACD a-[batch, 1], 当前a-[batch, 300]
   

        q_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        # ------------------------------------------ Train Actor ----------------------------------------#
        for params in self.q_critic.parameters():
            # Freeze Q net, so you don't waste time on computing its gradient while updating Actor.
            params.requires_grad = False

        probs = self.actor(s)  # [b,a_dim]
        log_probs = torch.log(probs + 1e-8)  # [b,a_dim]
        with torch.no_grad():
            q1_all, q2_all = self.q_critic(s)  # [b,a_dim]
        min_q_all = torch.min(q1_all, q2_all)

        a_loss = torch.sum(probs * (self.alpha * log_probs - min_q_all), dim=1, keepdim=True)  # [b,1]

        self.actor_optimizer.zero_grad()
        a_loss.mean().backward()
        self.actor_optimizer.step()

        for params in self.q_critic.parameters():
            params.requires_grad = True

        # ------------------------------------------ Train Alpha ----------------------------------------#
        if self.adaptive_alpha:
            with torch.no_grad():
                self.H_mean = -torch.sum(probs * log_probs, dim=1).mean()
            alpha_loss = self.log_alpha * (self.H_mean - self.target_entropy)

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp().item()

        # ------------------------------------------ Update Target Net ----------------------------------#
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def forward(self):
        return True
