# -*- encoding: utf-8 -*-
"""
@File    : select_video_quality.py
@Time    : 2023/11/24 10:50
@Author  : junruitian
@Software: PyCharm
"""
import numpy as np
import torch
from torch import nn
import os

class KFS_SVQ:
    def __init__(self, t):
        self.stress_detection_model = ""
        self.linear_array = [nn.Linear(768, 3).cuda() for i in range(0, t)]
        self.external_model = ExternalModel()
        # 模型载入路径
        model_dir = 'ubuntu/HDD/models/00001.pth'



    def get_external_model_result(self, video_rep, external_model):
        """
        接入外部模型获得结果的接口
        :param video_rep: 视频表示，形状为 (length, k)
        :param external_model: 外部模型对象
        :return: 外部模型的输出结果
        """
        video_rep = video_rep.cuda().float()
        with torch.no_grad():
            result = external_model(video_rep)
        return result
    
    def get_score(self, video_rep, t=100, alpha=130.0, r=0.88):
        """
        Calculates the quality score for a given video representation using T passes.
        Parameters
        ----------
        video_rep : shape (length, k)
            video representation from upstream task. tensor k dim.
        t : int, optional
            Amount of forward passes to use. The default is 100.
        alpha : float, optional
            Stretching factor, can be choosen to scale the score values
        r : float, optional
            Score displacement
        Returns
        -------
        score : float.
        """
        # Quality Process
        # video_rep : 300/100, 768
        video_rep = video_rep.cuda().float()

        for s_i in range(0, t):
            video_sequence = self.get_external_model_result(video_rep)
            video_sequence = video_sequence.view(-1)
            # print(video_sequence.shape)
            # shape [3*100/300] = [300, 900]
            input_blob = torch.unsqueeze(video_sequence, dim=0)
            if s_i == 0:
                repeated = input_blob
            else:
                repeated = torch.cat((repeated, input_blob), dim=0)
        # print(repeated.shape)
        # shape [t, 300/900]

        X = repeated
        norm = torch.nn.functional.normalize(X, dim=1)

        # Only get the upper triangle of the distance matrix

        # 计算向量欧氏距离
        eucl_dist_b = torch.cdist(norm, norm)
        # 取出上三角矩阵
        eucl_dist_b = torch.triu(eucl_dist_b)
        # 拉平向量
        eucl_dist_b = eucl_dist_b.reshape(eucl_dist_b.shape[0] * eucl_dist_b.shape[1])
        # 取出非零向量
        eucl_dist_b = eucl_dist_b[eucl_dist_b != 0]

        # Calculate score as given in the paper
        dis_mean = torch.mean(eucl_dist_b)
        score = 2 * (1 / (1 + torch.exp(dis_mean)))
        # 去掉tensor 梯度
        score = score.detach().cpu()
        # Normalize value based on alpha and r
        norm_res = 1 / (1 + np.exp(-(alpha * (score - r))))

        return norm_res

class ExternalModel(nn.Module):
    def __init__(self, model_dir):
        super(ExternalModel, self).__init__()
        self.model_dir = model_dir
        self.fc = nn.Linear(768, 1)

        # 随机选择一个模型文件路径
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        if not model_files:
            raise ValueError(f"No model files found in {model_dir}")
        random_model_file = random.choice(model_files)
        model_path = os.path.join(model_dir, random_model_file)

        # 加载模型参数
        try:
            self.load_state_dict(torch.load(model_path))
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")

    def forward(self, x):
        return self.fc(x)

class ExternalModel(nn.Module):
    def __init__(self, model_dir):
        super(ExternalModel, self).__init__()
        self.model_dir = model_dir
        # 假设模型是一个简单的线性层，你可以根据实际情况修改
        self.fc = nn.Linear(768, 1)

        # 随机选择一个模型文件路径
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        if not model_files:
            raise ValueError(f"No model files found in {model_dir}")
        random_model_file = random.choice(model_files)
        model_path = os.path.join(model_dir, random_model_file)

        # 加载模型参数
        try:
            self.load_state_dict(torch.load(model_path))
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")

    def forward(self, x):
        return self.fc(x)
