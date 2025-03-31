# -*- encoding: utf-8 -*-
"""
@File    : Config.py
@Time    : 2023/8/14 14:16
@Author  : junruitian
@Software: PyCharm
"""
import argparse
import warnings


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


def DefaultConfig():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', default="/home/HDD/junruit/MV_representation_100", type=str)
    # 下面这个参数需要加上，torch内部调用多进程时，会使用该参数，对每个gpu进程而言，其local_rank都是不同的；
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--distribute', default=False, type=bool)
    parser.add_argument('--use_gpu', default=True, type=bool)
    # Emotion_Attention for stress video detection
    parser.add_argument('--model_name', type=str, default='SACD_Agent')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')

    parser.add_argument('--save_directory', default="./exp/", type=str)
    parser.add_argument('--capacity', default=50000, type=int)  # replay buffer size

    parser.add_argument('--num_classes', default=3, type=int)  # 分类器
    parser.add_argument('--group_include_video', default=3, type=int)  # 分类器

    parser.add_argument('--mode_2', default="Graph", type=str)

    parser.add_argument('--total_frame_size', default=100, type=int, help="data input frames number for each video")
    parser.add_argument('--initial_state', default=0, type=int, help="parameter for initial 0 to select or 1 to delete")
    parser.add_argument('--max_episode_steps', default=2000, type=int, help="Max steps for RL Training")

    parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
    parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
    parser.add_argument('--ModelIdex', type=int, default=50, help='which model to load')
    parser.add_argument('--EnvIdex', type=int, default=0, help='which environment to load')
    parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')

    parser.add_argument('--seed', type=int, default=5, help='random seed')

    parser.add_argument('--x1', type=int, default=0.6, help='x1: The reward rate of quality')
    parser.add_argument('--x2', type=int, default=0.4, help='x2: The reward rate of frame nums')
    parser.add_argument('--t', type=int, default=100, help='video quality estimation parameter')

    parser.add_argument('--random_steps', type=int, default=1e4, help='steps for random policy to explore')
    parser.add_argument('--update_every', type=int, default=50, help='training frequency')
    parser.add_argument('--save_interval', type=int, default=1e4, help='Model saving interval, in steps.')
    parser.add_argument('--eval_interval', type=int, default=2e3, help='Model evaluating interval, in steps.')

    parser.add_argument('--alpha', type=float, default=0.2, help='init alpha')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--hid_shape', type=list, default=[200, 200], help='Hidden net shape')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive alpha turning')
    args = parser.parse_args()
    return args


def parse(self, kwargs):
    '''
    根据字典更新config参数
    :param kwargs:
    :return:
    '''
    # 更新配置参数
    for k, v in kwargs.items():
        if not hasattr(self, k):
            # 这里报错或者警告
            warnings.warn("Warning: opt has not attribute %s" % k)
        setattr(self, k, v)
    # 打印配置信息
    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


# DefaultConfig.parse = parse
opt = DefaultConfig()
