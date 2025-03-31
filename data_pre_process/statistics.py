# -*- encoding: utf-8 -*-
"""
@File    : statistics.py
@Time    : 2023/11/13 10:11
@Author  : junruitian
@Software: PyCharm
"""


def stat_stress_level():
    stress_level_dictionary = {}
    with open(file_path, 'r', encoding='utf-8', ) as fr:
        line = fr.readline()
        while line:
            stress_level = line.split(",")[10]
            if stress_level not in stress_level_dictionary:
                stress_level_dictionary[stress_level] = 1
            else:
                stress_level_dictionary[stress_level] = stress_level_dictionary[stress_level] + 1
            line = fr.readline()
    return stress_level_dictionary


def commit_valid(file):
    st_record = []
    with open(file, 'r', encoding='utf-8') as fr:
        content = fr.readline()
        while content:
            commit_id = content.split(",")[0]
            if commit_id not in st_record:
                st_record.append(commit_id)
            else:
                print(commit_id)
                print("Warning!")

            content = fr.readline()
    print(len(st_record))


def user_valid(file):
    st_record = []
    with open(file, 'r', encoding='utf-8') as fr:
        content = fr.readline()
        while content:
            user = content.split(",")[0]
            if user not in st_record:
                st_record.append(user)
            else:
                print(user)
                print("Warning!")

            content = fr.readline()
    print(len(st_record))


def number_gender(submit_file, user_file):
    submit_array = []
    with open(submit_file, 'r', encoding='utf-8') as fs:
        content = fs.readline()
        while content:
            submit_id = content.split(",")[12]
            submit_array.append(submit_id)
            content = fs.readline()
    # in file 69 user 27:41
    # actually 52 22:31
    male = 0
    female = 0
    with open(user_file, 'r', encoding='utf-8') as fr:
        content = fr.readline()
        while content:
            user_id = content.split(",")[0]
            if ("Male" in content) and (user_id in submit_array):
                male = male + 1
            elif ("Female" in content) and (user_id in submit_array):
                female = female + 1
            content = fr.readline()
    print(male)
    print(female)


def number_work(submit_file, user_file):
    submit_array = []
    with open(submit_file, 'r', encoding='utf-8') as fs:
        content = fs.readline()
        while content:
            submit_id = content.split(",")[12]
            submit_array.append(submit_id)
            content = fs.readline()
    work_array = {}
    grade_array = {}
    with open(user_file, 'r', encoding='utf-8') as fr:
        content = fr.readline()
        while content:
            work = content.split(",")[6]
            grade = content.split(",")[7]
            user_id = content.split(",")[0]
            if user_id in submit_array:
                if work in work_array:
                    work_array[work] = work_array[work] + 1
                else:
                    work_array[work] = 1
                if grade in grade_array:
                    grade_array[grade] = grade_array[grade] + 1
                else:
                    grade_array[grade] = 1

            content = fr.readline()
    print(work_array)
    print(grade_array)


def number_age(submit_file, user_file):
    submit_array = []
    with open(submit_file, 'r', encoding='utf-8') as fs:
        content = fs.readline()
        while content:
            submit_id = content.split(",")[12]
            submit_array.append(submit_id)
            content = fs.readline()
    age_array = {}
    with open(user_file, 'r', encoding='utf-8') as fr:
        content = fr.readline()
        while content:
            age_ = content.split(",")[5]
            user_id = content.split(",")[0]
            if user_id in submit_array:
                if age_ in age_array:
                    age_array[age_] = age_array[age_] + 1
                else:
                    age_array[age_] = 1
            content = fr.readline()
    print(age_array)


def number_submit(submit_file):
    submit_array = []
    with open(submit_file, 'r', encoding='utf-8') as fs:
        content = fs.readline()
        while content:
            submit_id = content.split(",")[12]
            submit_array.append(submit_id)
            content = fs.readline()


def ubuntu_length():
    import os
    length = []
    dataset_path = "/home/HDD/junruit/multi_view_dataset"
    user_list = os.listdir(dataset_path)
    for user in user_list:
        if (len(user)) == 8 and (user[0] != 'd'):
            # "/home/HDD/junruit/multi_view_dataset/12345678"
            user_data = os.path.join(dataset_path, user)
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
                        # "/home/HDD/junruit/multi_view_dataset/12345678/week1/99999999/sliver1111.mp4"
                        video_path = os.path.join(commit_path, video)
                        duration = get_video_duration(video_path)
                        with open("duration.txt", 'a+', encoding='utf-8') as fw:
                            fw.write(str(int(duration)))
                            fw.write("\n")
                        length.append(duration)


def get_video_duration(file_path):
    import cv2
    cap = cv2.VideoCapture(file_path)

    # 获取视频的帧数和帧率
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 计算视频时长
    duration = frame_count / fps

    cap.release()

    return duration


def cal_time_distribution():
    duration_path = "duration.txt"
    with open(duration_path, 'r', encoding='utf-8') as fr:
        content = fr.read()
    dura_ = [int(i) for i in content.split("\n") if i != ""]
    print(dura_)
    max_value = max(dura_)
    min_value = min(dura_)
    print(max_value)
    print(min_value)
    import numpy as np

    # 示例视频长度数组
    video_lengths = np.array(dura_)

    # 指定划分的范围
    ranges = [i for i in range(1, 1051, 50)]

    # 使用 np.digitize 将视频长度划分到范围
    range_indices = np.digitize(video_lengths, ranges)

    # 打印结果
    # 使用 np.histogram 划分区间，并统计每个区间中元素的数量
    hist, bin_edges = np.histogram(video_lengths, bins=ranges)

    # 打印每个区间对应的数组
    for i in range(len(bin_edges) - 1):
        range_array = video_lengths[(video_lengths >= bin_edges[i]) & (video_lengths < bin_edges[i + 1])]
        print(f"区间 {bin_edges[i]} - {bin_edges[i + 1]} 对应的数组：{range_array} 长度 {len(range_array)}")


def week_calculate():
    dataset_path = "/home/HDD/junruit/multi_view_dataset"
    duration = {}
    for i in range(1, 10):
        duration[str(i)] = []
    user_list = os.listdir(dataset_path)
    for user in user_list:
        if (len(user)) == 8 and (user[0] != 'd'):
            # "/home/HDD/junruit/multi_view_dataset/12345678"
            user_data = os.path.join(dataset_path, user)
            week_list = os.listdir(user_data)
            for i in range(1, 10):
                week_data = os.path.join(user_data, "week" + str(i))
                if not os.path.exists(week_data):
                    continue
                commit_list = os.listdir(week_data)
                for commit_id in commit_list:
                    # "/home/HDD/junruit/multi_view_dataset/12345678/week1/99999999/"
                    commit_path = os.path.join(week_data, commit_id)
                    video_list = os.listdir(commit_path)
                    for video in video_list:
                        # "/home/HDD/junruit/multi_view_dataset/12345678/week1/99999999/sliver1111.mp4"
                        video_path = os.path.join(commit_path, video)
                        duration_ = get_video_duration(video_path)
                        duration[str(i)].append(int(duration_))
    import json
    # 将字典写入文件
    with open('output.json', 'w') as file:
        json.dump(duration, file)


def week_avg():
    import json
    dataset_path = "output.json"
    with open(dataset_path, 'r') as file:
        data = json.load(file)
    sums = {key: sum(value) for key, value in data.items()}
    length = {key: len(value) for key, value in data.items()}
    print(sums)
    print(length)
    avg = []
    for i in range(1, 10):
        avg.append(sums[str(i)] / length[str(i)])
    print(avg)


def stress_change():
    for i in range(1, 10):
        stress = {}
        file_path = "F:\multi_view_dataset\database\\collectinfo_wk" + str(i) + ".txt"
        with open(file_path, 'r', encoding='utf-8') as fr:
            content = fr.readline()
            while content:
                stress_level = content.split(",")[10]
                user_id = content.split(",")[12]
                pro, grade = profess_get(user_id)
                if pro + grade in stress:
                    stress[pro + grade].append(int(stress_level))
                else:
                    stress[pro + grade] = [int(stress_level)]
                content = fr.readline()
        # print(stress)
        sums = {key: sum(value) for key, value in stress.items()}
        length = {key: len(value) for key, value in stress.items()}
        avg = {}
        for key, value in stress.items():
            avg[key] = round(sums[key] / length[key], 2)
        print(avg)
        # print(sums)
        # print(length)


def profess_get(user_id):
    file_path = "F:\multi_view_dataset\database\\user.txt"
    with open(file_path, 'r', encoding='utf-8') as fr:
        content = fr.readline()
        while content:
            if content.split(",")[0] == user_id:
                return content.split(",")[6], content.split(",")[7]
            else:
                content = fr.readline()


if __name__ == "__main__":
    import os

    if os.sep == '\\':
        # file_path = "F:\multi_view_dataset\database\\collectinfo.txt"
        # stress_level_dictionary = stat_stress_level()
        # print(stress_level_dictionary)
        # # commit_valid(file_path)
        # user_file = "F:\multi_view_dataset\database\\user.txt"
        # # user_valid(user_file)
        # number_gender(file_path, user_file)
        # number_age(file_path, user_file)
        # number_work(file_path, user_file)
        # cal_time_distribution()
        # week_avg()
        stress_change()
    else:
        # ubuntu_length()
        week_calculate()
