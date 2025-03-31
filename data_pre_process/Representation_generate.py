# -*- encoding: utf-8 -*-
"""
@File    : Representation_generate.py
@Time    : 2023/12/28 14:37
@Author  : junruitian
@Software: PyCharm
"""

import os

import numpy as np
from PIL import Image
from torch import nn
from transformers import ViTFeatureExtractor, ViTModel


class compared_pretrain(nn.Module):
    def __init__(self, path, number):
        super(compared_pretrain, self).__init__()

        self.model_name_or_path = 'google/vit-base-patch16-224-in21k'
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_name_or_path)
        self.vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.face_data_path = path
        self.face_align_number = number

    def forward(self):
        user_list = os.listdir(self.face_data_path)
        # img_path = "/home/HDD/junruit/MV_Face/12345678/week1/99999999/sliver1111/111.jpg"
        for user in user_list:
            # "/home/HDD/junruit/MV_Face/12345678"
            user_data = os.path.join(self.face_data_path, user)
            week_list = os.listdir(user_data)
            for week in week_list:
                # "/home/HDD/junruit/multi_view_dataset/12345678/week1"
                week_data = os.path.join(user_data, week)
                commit_list = os.listdir(week_data)
                for commit_id in commit_list:
                    # "/home/HDD/junruit/multi_view_dataset/12345678/week1/99999999/"
                    commit_path = os.path.join(week_data, commit_id)
                    video_list = os.listdir(commit_path)
                    for video in video_list:
                        # "/home/HDD/junruit/multi_view_dataset/12345678/week1/99999999/sliver1111"
                        video_path = os.path.join(commit_path, video)
                        if self.IsSuccess(video_path):
                            continue
                        else:
                            bool = self.video_transfer_representation(video_path)
                            if bool is True:
                                with open("Representation_generate_success.txt", 'a+', encoding='utf-8') as fw:
                                    fw.write(video_path)
                                    fw.write("\n")

        return True

    def IsSuccess(self, video_path):
        video = video_path.split(".")[0].split("/")[-1]
        with open("Representation_generate_success.txt", 'r', encoding='utf-8') as fr:
            content = fr.read()
        if video in content:
            return True
        else:
            return False

    def path_record(self, in_path, number):
        save_path = in_path.replace("MV_Face", "MV_representation_" + str(number))
        directory = ""
        for i in save_path.split("/")[0:-2]:
            directory = directory + i + "/"
        directory = directory + save_path.split("/")[-2]
        file = save_path.split("/")[-1].split(".")[0]
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory, file

    def face_align_num(self, list_img, num):
        array_length = len(list_img)
        img_list = sorted(list_img, key=lambda x: int(x.split(".")[-2].split("/")[-1]))

        stride = array_length // num
        selected_paths = []

        for i in range(0, array_length, stride):
            selected_paths.append(img_list[i])
        temp_length = len(selected_paths)
        if temp_length > 100:
            if (temp_length - 320) % 2 == 0:
                remove_list_forward = selected_paths[:(temp_length - 100) // 2]
                remove_list_back = selected_paths[-(temp_length - 100) // 2:]
            else:
                remove_list_forward = selected_paths[:(temp_length - 100) // 2]
                remove_list_back = selected_paths[-(temp_length - 100) // 2:]

            for i in remove_list_forward:
                selected_paths.remove(i)
            for j in remove_list_back:
                selected_paths.remove(j)
        if len(selected_paths) == num:
            return selected_paths
        else:
            assert len(selected_paths) == num, "length not right ! line 72"

    def video_transfer_representation(self, video_path):
        images = [os.path.join(video_path, i) for i in os.listdir(video_path)]
        images = self.face_align_num(images, self.face_align_number)
        for frame in images:
            directory, file = self.path_record(frame, self.face_align_number)
            print(directory, file)
            img = Image.open(frame)
            inputs = self.feature_extractor(img, return_tensors='pt')

            outputs = self.vit_model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            cls = last_hidden_states[:, 0, :]
            y = cls.detach().numpy()
            np.save(directory + "/" + file + ".npy", y)
            print(cls.shape)

        return True


if __name__ == "__main__":
    cur_sep = os.path.sep
    path = []
    face_data_path = "/home/HDD/junruit/MV_Face"
    num = 100
    Model = compared_pretrain(face_data_path, num)
    out = Model()
