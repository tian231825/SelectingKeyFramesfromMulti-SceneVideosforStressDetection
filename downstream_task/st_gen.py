# -*- encoding: utf-8 -*-
"""
@File    : st_gen.py
@Time    : 2024/4/7 17:41
@Author  : junruitian
@Software: PyCharm
"""
import torch
import numpy as np
# exp = "exp3_concat"
# for i in range(1, 27):
#     file = "../" + str(exp) + "/record_" + str(i) + ".txt"
#     loaded_np_array = np.loadtxt(file)
#     loaded_tensor = torch.FloatTensor(loaded_np_array)
#     count_ones = torch.sum(loaded_tensor == 1, dim=1)
#     reward_2 = count_ones / len(loaded_tensor[0])
#     print(reward_2)
#     if i == 1:
#         res_tensor = loaded_tensor
#     else:
#         res_tensor = torch.cat([res_tensor, loaded_tensor], dim=0)
#
# print(res_tensor.shape)

def commit_st():
    file = "../Configuration/collectinfo.txt"
    dict_ = {}
    with open(file, 'r', encoding='utf-8') as fr:
        line = fr.readline()
        while line:
            array = line.split(",")
            if array[12] not in dict_:
                dict_[array[12]] = 1
            else:
                dict_[array[12]] = dict_[array[12]] + 1
            line = fr.readline()
    st_ = {}
    for key,value in dict_.items():
        if str(value) not in st_:
            st_[str(value)] = 1
        else:
            st_[str(value)] = st_[str(value)] + 1

    print(st_)

def location_st():
    file = "../Configuration/collectinfo.txt"
    location = []
    with open(file, 'r', encoding='utf-8') as fr:
        line = fr.readline()
        while line:
            array = line.split(",")
            if array[1] not in location:
                location.append(array[1])
            if array[4] not in location:
                location.append(array[4])
            if array[7] not in location:
                location.append(array[7])
            line = fr.readline()


    print(location)
    print(len(location))


def event_st():
    file = "../Configuration/collectinfo.txt"
    event = []
    with open(file, 'r', encoding='utf-8') as fr:
        line = fr.readline()
        while line:
            array = line.split(",")
            if array[2] not in event:
                event.append(array[2])
            if array[5] not in event:
                event.append(array[5])
            if array[8] not in event:
                event.append(array[8])
            line = fr.readline()


    print(event)
    print(len(event))

if __name__ == "__main__":
    location_st()
    event_st()
