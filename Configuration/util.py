# -*- encoding: utf-8 -*-
"""
@File    : util.py
@Time    : 2023/11/9 19:39
@Author  : junruitian
@Software: PyCharm
"""
import argparse

import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, dvc, max_size=int(1e6)):
        self.max_size = max_size
        self.dvc = dvc
        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((max_size, state_dim), device=self.dvc)
        self.a = torch.zeros((max_size, state_dim), device=self.dvc)
        self.r = torch.zeros((max_size, 1), device=self.dvc)
        self.s_next = torch.zeros((max_size, state_dim), device=self.dvc)
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.dvc)

    def add(self, s, a, r, s_next, dw):
        import numpy as np
        if isinstance(s, np.ndarray):
            self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
        else:
            self.s[self.ptr] = s
        if isinstance(a, np.ndarray):
            self.a[self.ptr] = torch.from_numpy(a).to(self.dvc)
        else:
            self.a[self.ptr] = a
        self.r[self.ptr] = r
        # self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
        self.s_next[self.ptr] = s_next
        self.dw[self.ptr] = dw

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        ind = torch.randint(0, self.size, size=(1,), device=self.dvc)
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]


def evaluate_policy(env, model, video_a, video_b, video_c, turns=3):
    scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            # Take deterministic actions at test time
            a = model.select_action(s, deterministic=True)
            s_next, r, done, info = env.step(s, a, video_a, video_b, video_c)
            scores += r
            s = s_next

    return scores / turns


# You can just ignore 'str2bool'. Is not related to the RL.
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 'TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
