# -*- encoding: utf-8 -*-
"""
@File    : gpt_gen.py
@Time    : 2023/11/26 18:09
@Author  : junruitian
@Software: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class KeyframeSelectionNetwork(nn.Module):
    def __init__(self, num_videos, num_frames, num_features):
        super(KeyframeSelectionNetwork, self).__init__()

        # 图神经网络层
        self.gcn = GCNConv(num_features, num_features)

        # 全连接层
        self.fc1 = nn.Linear(num_videos * num_features, 256)
        self.fc2 = nn.Linear(256, num_videos * num_frames)

    def forward(self, videos):
        # videos: (batch_size, num_videos, num_frames, num_features)

        batch_size, num_videos, num_frames, num_features = videos.size()

        # 将每个视频的帧表示连接成一个图
        videos = videos.view(batch_size * num_videos, num_frames, num_features)
        videos = videos.transpose(1, 2).contiguous()
        edge_index = torch.tensor(
            [[i, i + 1] for i in range(0, batch_size * num_videos * num_frames, num_frames)]).t().contiguous()
        x = self.gcn(videos, edge_index)

        # 将每个视频的帧表示还原
        x = x.view(batch_size, num_videos, num_frames, num_features)

        # 池化操作，选择每个视频的关键帧
        pooled_frames, _ = torch.max(x, dim=2)

        # 全连接层
        x = F.relu(self.fc1(pooled_frames.view(batch_size, -1)))
        keyframes = torch.sigmoid(self.fc2(x))
        keyframes = keyframes.view(batch_size, num_videos, num_frames)

        return keyframes


'''
在这个示例中，我们定义了一个名为KeyframeSelectionNetwork的神经网络模型。
在初始化方法中，我们定义了图神经网络层GCNConv，以及两个全连接层fc1和fc2。
在forward方法中，我们首先将输入的视频帧表示连接成一个图，然后通过图神经网络层对帧表示进行图卷积操作。
接下来，我们将每个视频的帧表示还原，并使用最大池化操作选择每个视频的关键帧。
最后，我们通过两个全连接层进行特征提取和输出，最终得到三个视频对应的关键帧。
'''