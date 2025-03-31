# -*- encoding: utf-8 -*-
"""
@File    : statistic_aprf.py
@Time    : 2024/3/21 19:26
@Author  : junruitian
@Software: PyCharm
"""
import numpy as np


def three_dim_micro(confusion_matrix):
    # 计算总体真阳性、假阳性和假阴性
    TP = np.sum(np.diag(confusion_matrix))
    FP = np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    FN = np.sum(confusion_matrix, axis=1) - np.diag(confusion_matrix)

    # 计算微平均精确率、召回率和 F1 分数
    micro_precision = TP / (TP + np.sum(FP))
    micro_recall = TP / (TP + np.sum(FN))
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)

    # 打印结果
    print("Micro-average ACCURACY:", micro_f1)


def three_dim_macro(conf_matrix):
    # 计算每个类别的真阳性、假阳性和假阴性
    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP

    # 计算每个类别的精确率、召回率和 F1 分数
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)

    # 计算宏平均精确率、召回率和 F1 分数
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    # 打印结果
    print("Macro-average Precision:", macro_precision)
    print("Macro-average Recall:", macro_recall)
    print("Macro-average F1 Score:", macro_f1)


if __name__ == "__main__":
    # conf_matrix_all = np.array([[18, 20, 9],
    #                             [12, 15, 13],
    #                             [12, 9, 19]])
    # three_dim_macro(conf_matrix_all)
    # three_dim_micro(conf_matrix_all)
    # print("\n")
    # conf_matrix_exp1 = np.array([[25, 14, 8],
    #                              [14, 18, 8],
    #                              [16, 13, 11]])
    # three_dim_macro(conf_matrix_exp1)
    # three_dim_micro(conf_matrix_exp1)
    # print("\n")
    # conf_matrix_exp3 = np.array([[18, 20, 9],
    #                              [12, 15, 13],
    #                              [12, 9, 19]])
    # three_dim_macro(conf_matrix_exp3)
    # three_dim_micro(conf_matrix_exp3)
    # print("\n")
    # conf_matrix_all_2 = np.array([[29, 12, 6],
    #                              [6, 26, 8],
    #                              [3, 7, 30]])
    # three_dim_macro(conf_matrix_all_2)
    # three_dim_micro(conf_matrix_all_2)
    # print("\n")

    # # all-method
    # conf_matrix_all_2 = np.array([[27, 5, 6, 5, 2],
    #                               [3, 147, 17, 5, 2],
    #                               [4, 14, 189, 16, 4],
    #                               [4, 11, 14, 110, 5],
    #                               [2, 3, 2, 4, 34]])
    #
    # three_dim_micro(conf_matrix_all_2)
    # three_dim_macro(conf_matrix_all_2)
    # #  our method
    # conf_matrix_all_2 = np.array([[23, 5, 4, 7, 6],
    #                               [3, 145, 18, 6, 2],
    #                               [4, 13, 188, 17, 5],
    #                               [2, 6, 15, 118, 3],
    #                               [0, 3, 2, 1, 39]])
    #
    # three_dim_micro(conf_matrix_all_2)
    # three_dim_macro(conf_matrix_all_2)
    #
    # # Ada
    # conf_matrix_all_2 = np.array([[24, 6, 6, 5, 4],
    #                               [3, 148, 12, 9, 2],
    #                               [5, 15, 188, 15, 4],
    #                               [4, 10, 14, 112, 4],
    #                               [2, 3, 2, 2, 36]])
    #
    # three_dim_micro(conf_matrix_all_2)
    # three_dim_macro(conf_matrix_all_2)
    #
    # # GKEN
    # conf_matrix_all_2 = np.array([[20, 6, 7, 7, 5],
    #                               [2, 147, 14, 9, 2],
    #                               [4, 15, 190, 14, 4],
    #                               [5, 12, 11, 112, 4],
    #                               [3, 2, 1, 4, 35]])
    #
    # three_dim_micro(conf_matrix_all_2)
    # three_dim_macro(conf_matrix_all_2)
    #
    # # without GNN
    # conf_matrix_all_2 = np.array([[21, 7, 5, 6, 6],
    #                               [3, 141, 17, 10, 3],
    #                               [6, 15, 184, 17, 5],
    #                               [5, 12, 12, 111, 4],
    #                               [3, 3, 2, 5, 32]])
    #
    # three_dim_micro(conf_matrix_all_2)
    # three_dim_macro(conf_matrix_all_2)

    # print("Graph")
    # conf_matrix_all_2 = np.array([[175, 28, 16],
    #                              [14, 193, 20],
    #                              [17, 18, 154]])
    # three_dim_macro(conf_matrix_all_2)
    # three_dim_micro(conf_matrix_all_2)
    # print("\n")
    # print("Bi-LSTM")
    # conf_matrix_all_2 = np.array([[176, 26, 17],
    #                               [20, 189, 18],
    #                               [16, 23, 150]])
    # three_dim_macro(conf_matrix_all_2)
    # three_dim_micro(conf_matrix_all_2)
    # print("\n")
    # print("LSTM")
    # conf_matrix_all_2 = np.array([[175, 26, 18],
    #                              [17, 192, 18],
    #                              [18, 18, 153]])
    # three_dim_macro(conf_matrix_all_2)
    # three_dim_micro(conf_matrix_all_2)

    # # Our-Joint
    # conf_matrix_all_2 = np.array([[22,5,5,7,6],
    #                               [3,141,19,8,3],
    #                               [6,18,181,17,5],
    #                               [2,11,17,112,2],
    #                               [3,5,3,1,33]])
    # three_dim_micro(conf_matrix_all_2)
    # three_dim_macro(conf_matrix_all_2)
    #
    # # Our-Uniform
    # conf_matrix_all_2 = np.array([[22, 5, 5, 7, 6],
    #                               [3, 142, 18, 8, 3],
    #                               [7, 19,178,18,5],
    #                               [2,12,17,111,2],
    #                               [4, 4, 4, 1, 32]])
    # three_dim_micro(conf_matrix_all_2)
    # three_dim_macro(conf_matrix_all_2)

    # All-Joint
    conf_matrix_all_2 = np.array([[22, 10, 9, 2, 2],
                                  [7, 141, 18, 5, 3],
                                  [12, 25, 181, 6, 3],
                                  [3, 11, 13, 112, 5],
                                  [2, 4, 5, 2, 32]])
    three_dim_micro(conf_matrix_all_2)
    three_dim_macro(conf_matrix_all_2)

    # All-uniform
    conf_matrix_all_2 = np.array([[22, 9, 10, 2, 2],
                                  [8, 140, 15, 8, 3],
                                  [12, 28, 178, 6, 3],
                                  [4, 11, 13, 112, 4],
                                  [2, 5, 5, 2, 31]])
    three_dim_micro(conf_matrix_all_2)
    three_dim_macro(conf_matrix_all_2)

    # conf_matrix_all_2 = np.array([[28, 6, 6, 3, 2],
    #                               [7, 155, 6, 4, 2],
    #                               [19, 41, 153, 10, 4],
    #                               [8, 25, 9, 96, 6],
    #                               [5, 12, 5, 1, 22]])
    # three_dim_micro(conf_matrix_all_2)
    # three_dim_macro(conf_matrix_all_2)
    # conf_matrix_all_2 = np.array([[23, 8, 10, 2, 2],
    #                               [7, 142, 14, 8, 3],
    #                               [11, 26, 182, 5, 3],
    #                               [4, 11, 13, 113, 3],
    #                               [1, 5, 5, 2, 32]])
    # three_dim_micro(conf_matrix_all_2)
    # three_dim_macro(conf_matrix_all_2)
    # conf_matrix_all_2 = np.array([[23,4,5,7,6],
    #                               [3,144,17,7,3],
    #                               [7,19,180,16,5],
    #                               [2,10,17,114,1],
    #                               [3,4,4,1,33]])
    # three_dim_micro(conf_matrix_all_2)
    # three_dim_macro(conf_matrix_all_2)
    # conf_matrix_all_2 = np.array([[213., 4., 2.],
    #                               [91,134,2.],
    #                               [89,12,88.]])
    # three_dim_micro(conf_matrix_all_2)
    # three_dim_macro(conf_matrix_all_2)
    # conf_matrix_all_2 = np.array([[169,31,19],
    #                               [25,178,24],
    #                               [15,20,154]])
    # three_dim_micro(conf_matrix_all_2)
    # three_dim_macro(conf_matrix_all_2)
    # conf_matrix_all_2 = np.array([[172,27,20],
    #                               [26,182,19],
    #                               [17,20,152]])
    # three_dim_micro(conf_matrix_all_2)
    # three_dim_macro(conf_matrix_all_2)
    # conf_matrix_all_2 = np.array([[173, 27, 19],
    #                               [18, 190, 19],
    #                               [19, 19, 151]])
    # three_dim_micro(conf_matrix_all_2)
    # three_dim_macro(conf_matrix_all_2)
