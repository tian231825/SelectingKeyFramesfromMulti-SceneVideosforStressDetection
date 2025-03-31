# -*- encoding: utf-8 -*-
"""
@File    : statistics.py
@Time    : 2024/3/5 10:48
@Author  : junruitian
@Software: PyCharm
"""


def file_list():
    # time_1 = [t for t in range(415, 424)]
    # time_2 = [t for t in range(500, 509)]
    # time_3 = [t for t in range(521, 524)]
    # time_4 = [t for t in range(600, 611)]
    # time = time_1 + time_2 + time_3 + time_4
    # file_name = ["2024030" + str(t) + ".txt" for t in time]
    time_1 = [t for t in range(511, 524)]
    time_2 = [t for t in range(600, 613)]
    time = time_1 + time_2
    file_name = ["../exp3_concat/2024031" + str(t) + ".txt" for t in time]
    return file_name


def file_exp3():
    path = "../exp3_concat"
    import os
    file_name = [os.path.join(path, file) for file in os.listdir(path)]
    return file_name

def read_file(index):
    file_name = file_list()
    # print(file_name)
    effect_res, rate_res = [], []
    for file in file_name:
        e_a, r_a = collect_dat(file, index)
        # print(r_a)
        effect_res = effect_res + e_a
        rate_res = rate_res + r_a
    return effect_res, rate_res


# index is in batch [0,23]
def collect_dat(file, index=0):
    with open(file, 'r', encoding='utf-8') as fr:
        content = fr.read()
    data_in_batch = content.split("batch\n")
    data_in_line = [i.split("\n") for i in data_in_batch]
    data_in_line_filter = [element for element in data_in_line if element != ['']]
    effect_array = []
    rate_array = []
    for key in data_in_line_filter:
        x = key[index]
        effect_, rate_ = fetch_float(x)
        effect_array.append(effect_)
        rate_array.append(rate_)
    # print(rate_array)
    return effect_array, rate_array


def fetch_float(str_line):
    array = str_line.split(" ")
    print(array)
    if array[0][-1] == "8":
        print(array)
        effect, rate = float(array[0][0:5]), float(array[1][0:4])
    else:
        effect, rate = float(array[0][0:5]) * 10, float(array[1][0:4])
    # print(effect, rate)
    return effect, rate


# 单个视频的线
def step_rate_plot(a, b, c=None):
    import matplotlib.pyplot as plt
    # 创建图表
    plt.plot(a, b, marker='o', linestyle='-', color='b')
    if c != None:
        plt.plot(a, c, marker='o', linestyle='-', color='r')

    plt.gca().invert_xaxis()
    # 设置图表标题和轴标签
    plt.title('Line Plot of (a, b)')
    plt.xlabel('a')
    plt.ylabel('b')
    plt.savefig('line_plot.png')
    # 显示图表
    plt.show()


def sort_max(a, b):
    import numpy as np
    # 使用字典来存储每个相同的 a 对应的最大的 b
    max_b_dict = {}
    for ai, bi in zip(a, b):
        if ai not in max_b_dict or bi > max_b_dict[ai]:
            max_b_dict[ai] = bi

    # 得到排序后的 a 和对应的最大的 b
    sorted_a = np.array(list(max_b_dict.keys()))
    sorted_b = np.array(list(max_b_dict.values()))

    # 对结果按照 a 进行排序
    sort_order = np.argsort(sorted_a)
    sorted_a = sorted_a[sort_order]
    sorted_b = sorted_b[sort_order]

    # print("Original Arrays:")
    # print("a:", a)
    # print("b:", b)
    # print("\nResult after Operation:")
    # print("Sorted a:", sorted_a)
    # print("Sorted b:", sorted_b)

    return sorted_a, sorted_b


def sort_min(a, b):
    import numpy as np
    # 使用字典来存储每个相同的 a 对应的最大的 b
    min_b_dict = {}
    for ai, bi in zip(a, b):
        if ai not in min_b_dict or bi < min_b_dict[ai]:
            min_b_dict[ai] = bi

    # 得到排序后的 a 和对应的最大的 b
    sorted_a = np.array(list(min_b_dict.keys()))
    sorted_b = np.array(list(min_b_dict.values()))

    # 对结果按照 a 进行排序
    sort_order = np.argsort(sorted_a)
    sorted_a = sorted_a[sort_order]
    sorted_b = sorted_b[sort_order]

    # print("Original Arrays:")
    # print("a:", a)
    # print("b:", b)
    # print("\nResult after Operation:")
    # print("Sorted a:", sorted_a)
    # print("Sorted b:", sorted_b)

    return sorted_a, sorted_b


def min_max_normalization(arr):
    min_val = min(arr)
    max_val = max(arr)
    return (arr - min_val) / (max_val - min_val)


def change():
    a = [0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47,
         0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66,
         0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85,
         0.86, 0.87, 0.88, 0.89]
    b = [9.854000000000001, 12.536666666666667, 9.341333333333333, 16.250428571428575, 12.414250000000001,
         13.350000000000001, 11.3446, 15.239800000000002, 13.687166666666664, 11.020000000000001, 8.624500000000001,
         15.108999999999996, 12.835, 11.673714285714286, 13.16025, 11.728181818181817, 13.500357142857142,
         14.120428571428574, 13.257956521739128, 14.82518181818182, 14.899500000000002, 15.005208333333336,
         14.941250000000002, 15.343652173913043, 15.443333333333333, 15.365208333333335, 15.761833333333334, 15.9095,
         15.661583333333333, 15.775125000000001, 15.836958333333333, 15.985124999999998, 16.20516666666667,
         16.235583333333334, 16.313374999999997, 16.301958333333335, 16.479916666666668, 16.531291666666664,
         16.629833333333334, 16.820958333333333, 16.726291666666672, 16.612624999999994, 16.956416666666673,
         16.91720833333334, 17.036749999999994, 17.103375000000003, 17.270708333333335, 17.165125000000003,
         17.15441666666667, 17.280625000000004, 17.371791666666663, 17.36104166666667, 17.470375, 17.252041666666667,
         17.217874999999996, 17.678949999999997, 17.861529411764703, 17.117, 17.05, 16.88, 17.940000000000001]
    c = [(1 - A) for A in a]
    d = [(50 + B*2) for B in b]
    # print(c)
    return c, d

if __name__ == "__main__":
    dict = {}
    for index in range(0, 26):
        # 计算单条视频的趋势
        effect_res, rate_res = read_file(index)
        rate_best, effect_best = sort_max(rate_res, effect_res)
        for i in range(0, len(rate_best)):
            if rate_best[i] not in dict:
                dict[rate_best[i]] = [effect_best[i], 1]
            else:
                dict[rate_best[i]] = [
                    (dict[rate_best[i]][0] * dict[rate_best[i]][1] + effect_best[i]) / (dict[rate_best[i]][1] + 1),
                    (dict[rate_best[i]][1] + 1)]
    # rate_min, effect_min = sort_min(rate_res, effect_res)
    line_x = []
    line_y = []
    for key, value in dict.items():
        line_x.append(key)
        line_y.append(value[0])
    print(len(line_x), len(line_y))
    sorted_a_b = sorted(zip(line_x, line_y), key=lambda x: x[0])
    # 分离排序后的a和bprint(c)
    sorted_a, sorted_b = zip(*sorted_a_b)
    print(sorted_a)
    print(sorted_b)
    sorted_a, sorted_b = change()
    step_rate_plot(sorted_a, sorted_b)
