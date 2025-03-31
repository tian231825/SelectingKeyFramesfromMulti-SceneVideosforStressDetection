# -*- encoding: utf-8 -*-
"""
@File    : KFS_Module.py
@Time    : 2023/8/18 10:29
@Author  : junruitian
@Software: PyCharm
"""
import gym
import numpy as np
import torch
from gym import spaces
from gym.utils import seeding

from Config import opt
from .select_video_quality import KFS_SVQ

'''
Discrete： numbered from 0 to n-1. 举例， Discrete(n=4) 表示4个action上下左右。如果需要负数Discrete(3, start=-1)
Box：在 [low，high] 区间内的n维tensor。 举例，Box(low=0.0, high=255, shape=(210,160,3), dtype=np.uint8) 表示3Dtensor with 100800 bytes。
MultiBinary: n-shape的binary space。举例，MultiBinary(5) 表示5维的0或1的数组。 MultiBinary([3,2]) 表示3x2维的0或1的数组。
MultiDiscrete：一系列离散的action space。举例，MultiDiscrete([5,2,2]) 表示三个discrete action space。
Tuple：用于combine一些space instance。举例，Tuple(spaces=(Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32), Discrete(n=3), Discrete(n=2))).
Dict：也是用于combine一些space instance。 举例，Dict({'position':Discrete(2), 'velocity':Discrete(3)})
'''


class KFS_Environment(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, total_frame_size, initial_state=0, group_video=3):
        super(KFS_Environment, self).__init__()
        self.environment_name = "KFS_Environment"
        self.total_frame_size = total_frame_size
        self.group_video = group_video
        if initial_state == 0:
            self.initial_select_state = np.array([0] * self.total_frame_size)
        else:
            self.initial_select_state = np.array([1] * self.total_frame_size)

        # self.observation_space = spaces.Box(low=self.choose_min, high=self.choose_max, dtype=np.float32)
        self.observation_space = spaces.MultiDiscrete([2] * self.total_frame_size * self.group_video)  # select or not
        self.observation_low = np.array([0] * self.total_frame_size * self.group_video)
        self.observation_high = np.array([1] * self.total_frame_size * self.group_video)

        # self.action_space = spaces.Box(low=self.action_min, high=self.choose_max, dtype=np.float32)
        self.action_low = np.array([-1] * self.total_frame_size * self.group_video)
        self.action_high = np.array([1] * self.total_frame_size * self.group_video)
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=int)  # add delete stay

        self.seed()

        self.VQE = KFS_SVQ(opt.t)

    def seed(self, seed=None):
        '''
            step: 方法给出 terminated 或 truncated 信号后，调用 reset 启动新轨迹
        '''

        # 通过 super 初始化并使用基类的 self.np_random 随机数生成器
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, state, action, video_1, video_2, video_3):
        if isinstance(action, torch.Tensor):
            last_action = action.cpu()
            last_action = np.array(last_action)
        else:
            last_action = np.array(action)
        if isinstance(state, torch.Tensor):
            state = state.cpu()
        last_state = np.array(state)
        new_state = last_state + last_action
        # np.clip(u,0,1): u是一个一维数组* frame_size
        # 此元素若小于0， 则将0赋给此元素，若大于 1，则将1赋给此元素，
        # 若是在中间，则不变，作用是将动作值限定在[0,1]之间
        new_state = np.clip(new_state, 0, 1)
        new_state = torch.tensor(new_state).cuda()

        if opt.mode_2 == "concat":
            # torch.Size([batch, 100, 768]) --- torch.Size([batch, 300, 768])
            videos_rep = torch.cat((video_1, video_2, video_3), dim=1)
            # print(videos_rep.shape)
            # print("new state:", new_state.shape) # batch, 300
            # print("video rep:", videos_rep.shape) batch, 300, 768
            reward_1 = self.reward_calculate(new_state, videos_rep).cuda()

        elif opt.mode_2 == "Graph":
            reward_1_1 = self.reward_calculate(new_state[:, :100], video_1).cuda()
            reward_2_1 = self.reward_calculate(new_state[:, 100:200], video_2).cuda()
            reward_3_1 = self.reward_calculate(new_state[:, 200:], video_3).cuda()
            reward_1 = reward_1_1 + reward_2_1 + reward_3_1

        # reward_2 计算张量中1的数量作为frames
        # print(new_state)
        count_ones = torch.sum(new_state == 1, dim=1)
        reward_2 = count_ones / len(new_state[0])
        # 计算一维tensor的长度
        # reward 1 [24], reward 2 [24]
        # print("Reward:1,2", reward_1, reward_2)
        self.record_reward(reward_1, reward_2)
        average_value = torch.mean(reward_2)
        if average_value.item() <= 0.15:
            done = True
        # if reward_1 >= 0.8 and reward_2 <= 0.15:
        #     done = True
        # else:
        #     done = False
        else:
            done = False
        reward = 100 * reward_2 - 1e27 * reward_1
        # print(reward)
        # 提供附加信息，例如调试信息
        info = {'debug_info': 'Some additional information about the environment.'}
        return new_state, reward, done, info

    def reward_calculate(self, state, video):
        select_representation = state.unsqueeze(-1) * video  # 点乘 state 和 video representation
        # print(select_representation.shape) # [24, 300, 768]
        for idx in range(0, select_representation.shape[0]):
            score_ = self.VQE.get_score(select_representation[idx])
            score = torch.tensor([score_])

            if idx == 0:
                score_tensor = score
            else:
                score_tensor = torch.cat((score_tensor, score), dim=0)
        # score in [0, 1]
        # print(score_tensor.shape) [batch]
        return score_tensor

    def render(self, mode="human"):
        print("Step")

    def reset(self, batch=1, seed=None, input=None):
        '''
        step方法给出terminated或者truncated信号后，调用reset启动新轨迹
        :param seed:
        :return:
        '''
        # 通过super初始化并使用基类的self.np_random随机数生成器
        # super().reset(seed)
        if input is None:
            for idx in range(0, batch):
                state_ = self.np_random.uniform(size=self.total_frame_size * self.group_video)

                new_dimension_array = np.expand_dims(state_, axis=0)  # (1, 300)
                if idx == 0:
                    self.state = new_dimension_array
                else:
                    # print(self.state.shape)
                    self.state = np.concatenate((self.state, new_dimension_array), axis=0)
            self.last_action = None
        else:
            self.state = input
        # self.last_action = None
        info = {'debug_info': 'Reset Process in line 112, KFS Module.py'}
        return self.state, info

    def close(self):
        return

    def record_reward(self, r1, r2):
        from datetime import datetime
        current_time = datetime.now()
        current_time_str = current_time.strftime("%Y%m%d%H")
        name = current_time_str + ".txt"
        with open(name, 'a+', encoding='utf-8') as fw:
            for idx in range(0, r1.shape[0]):
                fw.write(str(r1[idx].item()))
                fw.write(" ")
                fw.write(str(round(r2[idx].item(), 4)))
                fw.write("\n")
            fw.write("batch\n")
