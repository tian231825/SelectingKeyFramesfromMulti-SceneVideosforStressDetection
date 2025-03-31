# -*- encoding: utf-8 -*-
"""
@File    : data_loader.py
@Time    : 2023/8/19 16:27
@Author  : junruitian
@Software: PyCharm
"""
import os
import random

import numpy as np
import torch
from torch.utils import data

from Config import opt


class Ada_Encoder(data.Dataset):
    def __init__(self, root, train=False, test=False, all=False):
        random.seed(opt.seed)
        self.commit_list_all = self.fetch_video_list(root=root)

        # self.commit_length = len(self.commit_list_all)
        self.commit_length = len(self.commit_list_all)
        self.test = test
        self.train = train
        self.all = all
        self.label_record = self.label_record_gen()
        if self.test:
            self.commits = self.commit_list_all[int(0.8 * self.commit_length):]
        elif self.train:
            self.commits = self.commit_list_all[:int(0.8 * self.commit_length)]
        if self.all:
            self.commits = self.commit_list_all

    def label_record_gen(self):
        label_record = {}
        with open("./Configuration/collectinfo.txt", 'r', encoding='utf-8') as fr:

            content = fr.readline()
            while content:
                index_id = content.split(",")[0]
                label = content.split(",")[10]
                if index_id not in label_record:
                    label_record[index_id] = label
                else:
                    print("Detection Repeat label && Index with Commit ID ", index_id)
                    assert 1 == -1, "Wrong label happen in dataset.py, Line 39 '{}' commit\n".format(index_id)
                content = fr.readline()
        return label_record

    def fetch_video_list(self, root):
        commit_list_all = []
        # F:\multi_view_dataset\\id_user\\week_i\\commit_id\\video1-2-3
        user_list = [os.path.join(root, user) for user in os.listdir(root)]
        for user in user_list:
            week_user_list = [os.path.join(user, week) for week in os.listdir(user)]
            for week_id in week_user_list:
                commit_list = os.listdir(week_id)
                for commit in commit_list:
                    commit_id = os.path.join(week_id, commit)
                    commit_list_all.append(commit_id)
        commit_sort = sorted(commit_list_all, key=lambda x: int(x.split("/")[-1]))
        # print(len(commit_sort))
        return commit_sort

    def __getitem__(self, item):

        commit = self.commits[item]

        video_order = sorted(os.listdir(commit), key=lambda x: int(x[10:18]))
        video_list = [os.path.join(commit, video) for video in video_order]

        video_embeddings = [self.Image_embedding_to_Video_tensor(video) for video in video_list]
        # print(len(video_embeddings))
        commit_id = commit.split("/")[-1]
        label = int(self.label_record[commit_id])
        index = item
        self.temp_record(video_list)
        
        return video_embeddings, label, index

    def Image_embedding_to_Video_tensor(self, video_path):
        frames = [os.path.join(video_path, img) for img in os.listdir(video_path)]
        frames = sorted(frames, key=lambda x: int(x.split("/")[-1].split(".")[-2]))
        for frame_id in range(0, len(frames)):
            '''
            frames[frame_id] :
            /home/HDD/junruit/fine_tune_vit/stressed_2_1.avi/988.npy
            '''

            frame_tensor = np.load(frames[frame_id], mmap_mode='r')
            frame_tensor = torch.from_numpy(frame_tensor)
            # print(frame_tensor.shape)
            # print(type(frame_tensor))
            '''
            每一帧的shape是(1, 768) 所以不需要stack 直接concat
            <class 'torch.Tensor'> torch.Size([1, 768])
            '''
            if frame_id == 0:
                video_tensor = frame_tensor
            else:
                video_tensor = torch.cat([video_tensor, frame_tensor], dim=0)

        # <class 'torch.Tensor'> torch.Size([100, 768])
        # print(video_tensor.shape)

        return video_tensor

    def temp_record(self, com):
        with open("video_list.txt", 'a+', encoding='utf-8') as fw:
            for i in com:
                fw.write(str(i))
            fw.write("\n")

    def __len__(self):
        return len(self.commits)
