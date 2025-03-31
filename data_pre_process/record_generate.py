# -*- encoding: utf-8 -*-
"""
@File    : record_generate.py
@Time    : 2023/10/15 10:19
@Author  : junruitian
@Software: PyCharm
"""

week = 8
collect_info_path = "F:\\multi_view_dataset\\database\\collectinfo_wk" + str(week) + ".txt"
user_info_path = "F:\\multi_view_dataset\\database\\user.txt"

submit_stat_list = "submit_stat_week_" + str(week) + ".csv"

with open(collect_info_path, 'r', encoding='utf-8') as fr:
    record = fr.read()
record_list = record.split("\n")[:-1]

with open(user_info_path, 'r', encoding='utf-8') as fr:
    user = fr.read()
user_list = user.split("\n")[:-1]

statistics_list = []
for user in user_list:

    temp_list = []
    user_information = user.split(",")
    for table in user_information:
        temp_list.append(table)
    for record in record_list:
        if user_information[0] == record.split(",")[12]:
            temp_list.append(record.split(",")[0])
            temp_list.append(record.split(",")[13])

    statistics_list.append(temp_list)

with open(submit_stat_list, 'w', encoding='utf-8') as fw:
    for st in statistics_list:
        for i in st:
            fw.write(str(i))
            fw.write(",")
        fw.write("\n")

