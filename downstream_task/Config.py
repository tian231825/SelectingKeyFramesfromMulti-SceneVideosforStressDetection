# -*- encoding: utf-8 -*-
"""
@File    : Config.py
@Time    : 2024/3/18 16:49
@Author  : junruitian
@Software: PyCharm
"""

import argparse


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
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--num_classes', default=3, type=int)  # 分类器
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--mode', default="train", type=str)
    parser.add_argument('--method', default="mlp", type=str)
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM bi or not')
    parser.add_argument('--seed', type=int, default=5, help='random seed')

    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--hid_shape', type=list, default=[200, 200], help='Hidden net shape')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')

    parser.add_argument('--record_dataloader', type=bool, default=True, help='record video list or not')
    args = parser.parse_args()
    return args


# DefaultConfig.parse = parse
opt = DefaultConfig()
