# -*- encoding: utf-8 -*-
"""
@File    : transformer_res.py
@Time    : 2024/3/8 12:57
@Author  : junruitian
@Software: PyCharm
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from Config import opt
from data_loader import data_Encoder


# 构建 Transformer 模型
class TransformerClassifier(nn.Module):
    def __init__(self, input_size, output_size, num_layers=6, hidden_size=512, num_heads=12, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=768, num_heads=12)  # 假设使用了 12 个头

        self.fc1 = nn.Linear(input_size, 500)
        self.fc2 = nn.Linear(500, output_size)

    def forward(self, x):
        self_attn_output, _ = self.self_attention(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))
        # 获取CLS token的输出，即第一个位置的输出
        cls_output = self_attn_output[0, :, :]  # 获取第一个位置的输出
        # print(cls_output.shape)
        output = self.fc1(cls_output)  # output.shape: (batch_size, 500)
        # print(output.shape)
        output = self.fc2(output)  # output.shape: (500, output_size)
        # print(output.shape)
        return output


class TransformerClassifier_2(nn.Module):
    def __init__(self, input_size, output_size, num_layers=6, hidden_size=512, num_heads=12, dropout=0.1):
        super(TransformerClassifier_2, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=768, num_heads=12)  # 假设使用了 12 个头

        self.fc1 = nn.Linear(768 * 3, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 768)
        self.fc3 = nn.Linear(input_size, 500)
        self.fc4 = nn.Linear(500, output_size)

    def forward(self, x):
        v1, v2, v3 = x[:, 0:100, :], x[:, 100:200, :], x[:, 200:300, :],
        self_attn_output_1, _ = self.self_attention(v1.transpose(0, 1), v1.transpose(0, 1), v1.transpose(0, 1))
        self_attn_output_2, _ = self.self_attention(v2.transpose(0, 1), v2.transpose(0, 1), v2.transpose(0, 1))
        self_attn_output_3, _ = self.self_attention(v3.transpose(0, 1), v3.transpose(0, 1), v3.transpose(0, 1))
        # 获取CLS token的输出，即第一个位置的输出
        cls_output_1 = self_attn_output_1[0, :, :]  # 获取第一个位置的输出
        cls_output_2 = self_attn_output_1[0, :, :]  # 获取第一个位置的输出
        cls_output_3 = self_attn_output_1[0, :, :]  # 获取第一个位置的输出
        # print(cls_output.shape)

        concatenated_tensor = torch.cat((cls_output_1, cls_output_2, cls_output_3), dim=1)
        out = self.fc1(concatenated_tensor)
        out = self.relu(out)
        out = self.fc2(out)
        output = self.fc3(out)  # output.shape: (batch_size, 500)
        # print(output.shape)
        output = self.fc4(output)  # output.shape: (500, output_size)
        # print(output.shape)
        return output


class TransformerClassifier_3(nn.Module):
    def __init__(self, input_size, output_size, num_layers=6, hidden_size=512, num_heads=12, dropout=0.1):
        super(TransformerClassifier_3, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=768, num_heads=12)  # 假设使用了 12 个头

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=opt.bidirectional)
        if opt.bidirectional:
            self.fc1 = nn.Linear(2*hidden_size, hidden_size)
        else:
            self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 5)
        self.fc4 = nn.Linear(15, output_size)

    def forward(self, x):
        v1, v2, v3 = x[:, 0:100, :], x[:, 100:200, :], x[:, 200:300, :],
        self_attn_output_1, _ = self.self_attention(v1.transpose(0, 1), v1.transpose(0, 1), v1.transpose(0, 1))
        self_attn_output_2, _ = self.self_attention(v2.transpose(0, 1), v2.transpose(0, 1), v2.transpose(0, 1))
        self_attn_output_3, _ = self.self_attention(v3.transpose(0, 1), v3.transpose(0, 1), v3.transpose(0, 1))
        # 获取CLS token的输出，即第一个位置的输出
        cls_output_1 = self_attn_output_1[0, :, :]  # 获取第一个位置的输出
        cls_output_2 = self_attn_output_1[0, :, :]  # 获取第一个位置的输出
        cls_output_3 = self_attn_output_1[0, :, :]  # 获取第一个位置的输出
        # print(cls_output_1.shape)

        concatenated_tensor = torch.stack([cls_output_1, cls_output_2, cls_output_3], dim=1)
        # print(concatenated_tensor.shape)
        output, (ht, ct) = self.lstm(concatenated_tensor)
        # print(output.shape)
        out = self.fc1(output)
        out = self.relu(out)
        out = self.fc2(out)
        out = out.view(out.shape[0], -1)

        output = self.fc4(out)  # output.shape: (500, output_size)
        # print(output.shape)
        return output


if __name__ == "__main__":
    # 初始化模型、损失函数和优化器

    # all_data = data_Encoder(opt.data_root, all=True)

    # all_dataloader = DataLoader(all_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # file = "label_train.txt"
    # test_data = data_Encoder(opt.data_root, train=True)
    # test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    # with open(file, 'w', encoding='utf-8') as fw:
    #     for i, (rep, label, index) in enumerate(test_dataloader):
    #         for v in label:
    #             fw.write(str(v.item()) + '\n')

    if opt.mode == "train":
        model = TransformerClassifier(input_size=768, output_size=opt.num_classes)  # 输入大小为 768，输出大小为 5
        model = model.cuda()
        criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器
        train_data = data_Encoder(opt.data_root, train=True)
        train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
        start_time = time.time()
        # 开始训练
        for epoch in range(opt.epoch):
            model.train()
            running_loss = 0.0
            for i, (rep, label, index) in enumerate(train_dataloader):
                video_a = rep[0]
                video_b = rep[1]
                video_c = rep[2]
                input_a = Variable(video_a)
                input_b = Variable(video_b)
                input_c = Variable(video_c)
                # label: torch.Size([batch])
                index = Variable(index)
                label = Variable(label)
                if opt.use_gpu:
                    input_a = input_a.cuda()
                    input_b = input_b.cuda()
                    input_c = input_c.cuda()
                    index = index.cuda()
                    label = label.cuda()
                videos_rep = torch.cat((input_a, input_b, input_c), dim=1)

                optimizer.zero_grad()
                outputs = model(videos_rep)  # 添加批次维度，因为模型输入期望是 (batch_size, seq_len, input_size)

                # print(outputs)
                # print(label)
                loss = criterion(outputs, label)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新权重

                running_loss += loss.item()
            print(f"Epoch {epoch + 1}/{opt.epoch}, Loss: {running_loss}")

        print("Training finished.")
        end_time = time.time()
        # 保存模型
        pth_m = "transformer_model_concat_" + str(opt.num_classes) + "_" + str(opt.epoch) + ".pth"
        torch.save(model.state_dict(), pth_m)
        duration = end_time - start_time
        print(duration)
    elif opt.mode == "train_2":
        model = TransformerClassifier_2(input_size=768, output_size=opt.num_classes)  # 输入大小为 768，输出大小为 5
        model = model.cuda()
        criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器
        train_data = data_Encoder(opt.data_root, train=True)
        train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
        start_time = time.time()
        # 开始训练
        for epoch in range(opt.epoch):
            model.train()
            running_loss = 0.0
            for i, (rep, label, index) in enumerate(train_dataloader):
                video_a = rep[0]
                video_b = rep[1]
                video_c = rep[2]
                input_a = Variable(video_a)
                input_b = Variable(video_b)
                input_c = Variable(video_c)
                # label: torch.Size([batch])
                index = Variable(index)
                label = Variable(label)
                if opt.use_gpu:
                    input_a = input_a.cuda()
                    input_b = input_b.cuda()
                    input_c = input_c.cuda()
                    index = index.cuda()
                    label = label.cuda()
                videos_rep = torch.cat((input_a, input_b, input_c), dim=1)

                optimizer.zero_grad()
                outputs = model(videos_rep)  # 添加批次维度，因为模型输入期望是 (batch_size, seq_len, input_size)

                # print(outputs)
                # print(label)
                loss = criterion(outputs, label)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新权重

                running_loss += loss.item()
            print(f"Epoch {epoch + 1}/{opt.epoch}, Loss: {running_loss}")

        print("Training finished.")
        end_time = time.time()
        # 保存模型
        pth_m = "transformer_model_mlp_" + str(opt.num_classes) + "_" + str(opt.epoch) + ".pth"
        torch.save(model.state_dict(), pth_m)
        duration = end_time - start_time
        print(duration)
    elif opt.mode == "train_3":
        model = TransformerClassifier_3(input_size=768, output_size=opt.num_classes)  # 输入大小为 768，输出大小为 5
        model = model.cuda()
        criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器
        train_data = data_Encoder(opt.data_root, train=True)
        train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
        start_time = time.time()
        # 开始训练
        for epoch in range(opt.epoch):
            model.train()
            running_loss = 0.0
            for i, (rep, label, index) in enumerate(train_dataloader):
                video_a = rep[0]
                video_b = rep[1]
                video_c = rep[2]
                input_a = Variable(video_a)
                input_b = Variable(video_b)
                input_c = Variable(video_c)
                # label: torch.Size([batch])
                index = Variable(index)
                label = Variable(label)
                if opt.use_gpu:
                    input_a = input_a.cuda()
                    input_b = input_b.cuda()
                    input_c = input_c.cuda()
                    index = index.cuda()
                    label = label.cuda()
                videos_rep = torch.cat((input_a, input_b, input_c), dim=1)

                optimizer.zero_grad()
                outputs = model(videos_rep)  # 添加批次维度，因为模型输入期望是 (batch_size, seq_len, input_size)

                # print(outputs)
                # print(label)
                loss = criterion(outputs, label)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新权重

                running_loss += loss.item()
            print(f"Epoch {epoch + 1}/{opt.epoch}, Loss: {running_loss}")

        print("Training finished.")
        end_time = time.time()
        # 保存模型
        pth_m = "transformer_model_bilstm" + str(opt.bidirectional) + "_" + str(opt.num_classes) + "_" + str(opt.epoch) + ".pth"
        torch.save(model.state_dict(), pth_m)
        duration = end_time - start_time
        print(duration)

    elif opt.mode == "test":
        if opt.method == "mlp":
            model = TransformerClassifier_2(input_size=768, output_size=opt.num_classes)  # 输入大小为 768，输出大小为 5
            method = opt.method
        elif opt.method == "lstm":
            model = TransformerClassifier_3(input_size=768, output_size=opt.num_classes)
            method = "bilstm" + str(opt.bidirectional)
        else:
            model = TransformerClassifier(input_size=768, output_size=opt.num_classes)  # 输入大小为 768，输出大小为 5
            method = opt.method
        model = model.cuda()

        # test_data = data_Encoder(opt.data_root, test=True)
        test_data = data_Encoder(opt.data_root, all=True)
        test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
        pth_file = "transformer_model_" + str(method) + "_" + str(opt.num_classes) + "_" + str(opt.epoch) + ".pth"
        # 加载模型进行测试
        model.load_state_dict(torch.load(pth_file))  # 加载训练好的模型参数
        model.eval()  # 设置为评估模式
        start_time = time.time()
        # 进行测试
        confusion_matrix = torch.zeros(opt.num_classes, opt.num_classes)
        with torch.no_grad():
            for i, (rep, label, index) in enumerate(test_dataloader):
                video_a = rep[0]
                video_b = rep[1]
                video_c = rep[2]
                input_a = Variable(video_a)
                input_b = Variable(video_b)
                input_c = Variable(video_c)
                # label: torch.Size([batch])
                label = Variable(label)
                index = Variable(index)
                if opt.use_gpu:
                    input_a = input_a.cuda()
                    input_b = input_b.cuda()
                    input_c = input_c.cuda()
                    label = label.cuda()
                    index = index.cuda()
                videos_rep = torch.cat((input_a, input_b, input_c), dim=1)

                outputs = model(videos_rep)  # [batch, num_classes]
                _, predicted = torch.max(outputs, 1)  # 获取预测结果
                for t, p in zip(label.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        end_time = time.time()
        duration = end_time - start_time
        print(duration)
        print(confusion_matrix)

    else:
        if opt.method == "mlp":
            model = TransformerClassifier_2(input_size=768, output_size=opt.num_classes)  # 输入大小为 768，输出大小为 5
            method = opt.method
        elif opt.method == "lstm":
            model = TransformerClassifier_3(input_size=768, output_size=opt.num_classes)
            method = "bilstm" + str(opt.bidirectional)
        else:
            model = TransformerClassifier(input_size=768,
                                          output_size=opt.num_classes)  # 输入大小为 768，输出大小为 5model = model.cuda()
            method = opt.method
        model = model.cuda()
        test_data = data_Encoder(opt.data_root, test=True)
        test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
        pth_file = "transformer_model_" + str(opt.method) + "_" + str(opt.num_classes) + "_" + str(opt.epoch) + ".pth"
        # 加载模型进行测试
        model.load_state_dict(torch.load(pth_file))  # 加载训练好的模型参数
        model.eval()  # 设置为评估模式

        # 进行测试
        confusion_matrix = torch.zeros(opt.num_classes, opt.num_classes)
        rate = np.array([0, 0, 0, 0])
        start_time = time.time()
        with torch.no_grad():
            for i, (rep, label, index, tensor_select) in enumerate(test_dataloader):
                video_a = rep[0]
                video_b = rep[1]
                video_c = rep[2]
                input_a = Variable(video_a)
                input_b = Variable(video_b)
                input_c = Variable(video_c)
                tensor_select = Variable(tensor_select)
                # label: torch.Size([batch])
                label = Variable(label)
                index = Variable(index)
                if opt.use_gpu:
                    input_a = input_a.cuda()
                    input_b = input_b.cuda()
                    input_c = input_c.cuda()
                    label = label.cuda()
                    index = index.cuda()
                    tensor_select = tensor_select.cuda()
                videos_rep = torch.cat((input_a, input_b, input_c), dim=1)
                # 计算张量中值为 1 的数量
                count_ones = torch.sum(torch.eq(tensor_select, 1))
                # 计算张量中大于0的数量
                sum_all_elements = torch.sum(tensor_select > 0)
                # 计算张量中大于0。5的数量
                count = torch.sum(tensor_select > 0.5)
                num_elements = tensor_select.numel()
                rate = rate + np.array([count_ones.item(), count.item(), sum_all_elements.item(), num_elements])
                # tensor_select = tensor_select.to(torch.double)
                # videos_rep = videos_rep.double()
                select_representation = tensor_select.unsqueeze(-1) * videos_rep
                # print(tensor_select.dtype)
                # print(videos_rep.dtype)
                # print(select_representation.dtype)
                outputs = model(select_representation)  # [batch, num_classes]
                _, predicted = torch.max(outputs, 1)  # 获取预测结果
                for t, p in zip(label.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        end_time = time.time()
        duration = end_time - start_time
        print(rate)
        print(confusion_matrix)
        print(duration)
