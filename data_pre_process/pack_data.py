# -*- encoding: utf-8 -*-
"""
@File    : pack_data.py
@Time    : 2023/10/14 16:56
@Author  : junruitian
@Software: PyCharm
"""
import os
import shutil
import time


def re_build_data(week):
    path_source = "G:\multi_view_dataset\\"
    read_file = "G:\multi_view_dataset\database\collectinfo_wk" + str(week) + ".txt"
    with open(read_file, 'r', encoding='utf-8') as fr:
        content = fr.readline()

        while content:
            data = content.split(",")
            id_commit = data[0]
            id_user = data[12]
            video_1 = data[3].split("/")[-1]
            video_2 = data[6].split("/")[-1]
            video_3 = data[9].split("/")[-1]
            if not os.path.exists(path_source + id_user):
                os.makedirs(path_source + id_user)
            path_week = path_source + id_user + "\\" + "week" + str(week) + "\\"
            if not os.path.exists(path_week):
                os.makedirs(path_week)
            path_commit = path_week + id_commit
            if not os.path.exists(path_commit):
                os.makedirs(path_commit)
            if os.path.exists(path_source + video_1):
                shutil.move(path_source + video_1, path_commit + "\\" + video_1)

            if os.path.exists(path_source + video_2):
                shutil.move(path_source + video_2, path_commit + "\\" + video_2)

            if os.path.exists(path_source + video_3):
                shutil.move(path_source + video_3, path_commit + "\\" + video_3)


            print(id_user + "," + id_commit)
            content = fr.readline()


if __name__ == "__main__":
    for week in range(2, 10):
        re_build_data(week)
