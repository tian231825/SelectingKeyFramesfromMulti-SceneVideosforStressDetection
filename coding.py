# -*- encoding: utf-8 -*-
"""
@File    : test_code.py
@Time    : 2023/11/13 18:47
@Author  : junruitian
@Software: PyCharm
"""


# import gym
# import numpy as np
# import gymnasium as gym

# class KFS_Environment(gym.Env):
#     """Custom Environment that follows gym interface"""
#     def __init__(self, total_frame_size, initial_state=0):
#         super(KFS_Environment, self).__init__()
#         self.environment_name = "KFS_Environment"
#         self.total_frame_size = total_frame_size
#
#         if initial_state == 0:
#             self.initial_select_state = np.array([0] * self.total_frame_size)
#         else:
#             self.initial_select_state = np.array([1] * self.total_frame_size)
#
#
#         self.choose_min = np.array([0] * self.total_frame_size)
#         self.choose_max = np.array([1.] * self.total_frame_size)
#
#         self.action_min = np.array([-1.] * self.total_frame_size)
#
#         # self.observation_space = spaces.Box(low=self.choose_min, high=self.choose_max, dtype=np.float32)
#         self.observation_space = spaces.MultiDiscrete([[2] * self.total_frame_size] * 3)  # select or not
#
#         # self.action_space = spaces.Box(low=self.action_min, high=self.choose_max, dtype=np.float32)
#         low = np.array([-1] * self.total_frame_size)
#         high = np.array([1] * self.total_frame_size)
#         self.action_space = spaces.Box(low, high, dtype=int)   # add delete stay
#
#         print(self.observation_space.shape[0])
#         print(self.action_space.seed(5))

#
# def ser_fiq(T=100, alpha=130.0, r=0.88):
#     from sklearn.metrics.pairwise import euclidean_distances
#     from sklearn.preprocessing import normalize
#     import torch
#     aligned_img = np.random.rand(3, 100, 150)
#
#     if aligned_img.shape[0] != 3:
#         aligned_img = np.transpose(aligned_img, (2, 0, 1))
#     print(aligned_img.shape)
#     input_blob = np.expand_dims(aligned_img, axis=0)
#     print(input_blob.shape)
#     repeated = np.repeat(input_blob, T, axis=0)
#     print(repeated.shape)
#     # gpu_repeated = mx.nd.array(repeated, ctx=self.device)
#     gpu_repeated = torch.tensor(repeated)
#     print(gpu_repeated.shape)
#
#     X = np.array(gpu_repeated.cpu())
#     norm = normalize(X, axis=1)
#
#     # Only get the upper triangle of the distance matrix
#     eucl_dist = euclidean_distances(norm, norm)[np.triu_indices(T, k=1)]
#
#     # Calculate score as given in the paper
#     score = 2 * (1 / (1 + np.exp(np.mean(eucl_dist))))
#     # Normalize value based on alpha and r
#     print(score)
#     return 1 / (1 + np.exp(-(alpha * (score - r))))
#
#
# def ser_fiq_gpu(T=100, alpha=130.0, r=0.88):
#     from sklearn.metrics.pairwise import euclidean_distances
#     from sklearn.preprocessing import normalize
#     import torch
#     aligned_img = torch.randn(3, 100, 150)
#     input_blob = torch.unsqueeze(aligned_img, dim=0)
#     print(input_blob.shape)
#     repeated = input_blob.repeat(T, 1, 1, 1)
#     print(repeated.shape)
#     # gpu_repeated = mx.nd.array(repeated, ctx=self.device)
#     gpu_repeated = torch.tensor(repeated)
#     print(gpu_repeated.shape)
#
#     X = np.array(gpu_repeated.cpu())
#     norm = normalize(X, axis=1)
#
#     # Only get the upper triangle of the distance matrix
#     eucl_dist = euclidean_distances(norm, norm)[np.triu_indices(T, k=1)]
#
#     # Calculate score as given in the paper
#     score = 2 * (1 / (1 + np.exp(np.mean(eucl_dist))))
#     # Normalize value based on alpha and r
#     print(score)
#     return 1 / (1 + np.exp(-(alpha * (score - r))))
#
#
# def norma():
#     import torch
#     from sklearn.metrics.pairwise import euclidean_distances
#     from sklearn.preprocessing import normalize
#     t = 8
#     a_img = torch.randn(t, 5)
#     b_img = np.random.rand(t, 5)
#
#     norm_a = normalize(b_img, axis=1)
#     norm_b = torch.nn.functional.normalize(a_img, dim=1)
#     print(norm_a.shape)
#     print(norm_b.shape)
#
#     # eucl_dist_a = euclidean_distances(norm_a, norm_a)[np.triu_indices(t, k=1)]
#     eucl_dist_a = euclidean_distances(norm_a, norm_a)
#     print(eucl_dist_a)
#     eucl_dist_a = eucl_dist_a[np.triu_indices(t, k=1)]
#     print(eucl_dist_a)
#
#
#     # eucl_dist_b = torch.cdist(norm_b, norm_b)[torch.triu_indices(row=t,col=t, offset=1)]
#
#     # 计算向量欧氏距离
#     eucl_dist_b = torch.cdist(norm_b, norm_b)
#     # 取出上三角矩阵
#     eucl_dist_b = torch.triu(eucl_dist_b)
#     # 拉平向量
#     eucl_dist_b = eucl_dist_b.reshape(eucl_dist_b.shape[0] * eucl_dist_b.shape[1])
#     # 取出非零向量
#     eucl_dist_b = eucl_dist_b[eucl_dist_b != 0]
#
#     dis_mean = torch.mean(eucl_dist_b)
#     score = 2 * (1 / (1 + torch.exp(dis_mean)))
#     print(dis_mean)
#     print(score)


# def rotate_image():
#
#     image = cv2.imread('8512.jpg')
#
#     # 获取图像中心点坐标
#     height, width = image.shape[:2]
#     center = (width // 2, height // 2)
#
#     # 设置旋转角度（正值为逆时针旋转）
#     angle = 90
#
#     # 生成旋转矩阵
#     rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
#
#     # 进行旋转变换
#     rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
#
#     # 显示原始图像和旋转后的图像
#     cv2.imshow('Original Image', image)
#     cv2.imshow('Rotated Image', rotated_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
def rotate_image(image_path):
    import cv2
    image = cv2.imread(image_path)
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow('Rotated Image', rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cal_ten():
    import torch

    # 创建长度为2的一维张量
    tensor1 = torch.tensor([2, 3])

    # 创建2x3的二维张量
    tensor2 = torch.tensor([[1, 2, 3],
                            [4, 5, 6]])

    # 使用广播进行逐元素相乘
    result_tensor = tensor1.view(-1, 1) * tensor2

    # 输出结果
    print("Tensor 1:")
    print(tensor1)
    print("\nTensor 2:")
    print(tensor2)
    print("\nResult Tensor (broadcasted multiplication):")
    print(result_tensor)


def asasaas():
    import torch
    import numpy as np
    # from sklearn.preprocessing import normalize
    t = 100

    video_rep = torch.randn(768)
    input_blob = torch.unsqueeze(video_rep, dim=0)
    line = [torch.nn.Linear(768, 500) for i in range(0, t)]
    for i in range(0, t):

        if i == 0:
            rep_500 = line[i](video_rep).unsqueeze(dim=0)
        else:
            rep = line[i](video_rep).unsqueeze(dim=0)
            rep_500 = torch.cat((rep_500, rep), dim=0)

    # repeated = input_blob.repeat(t, 1)
    X = rep_500
    print(X.shape)
    norm = torch.nn.functional.normalize(X, p=2, dim=1)
    print(norm)
    # Only get the upper triangle of the distance matrix

    # 计算向量欧氏距离
    eucl_dist_b = torch.cdist(norm, norm)
    # 取出上三角矩阵
    eucl_dist_b = torch.triu(eucl_dist_b)
    # 拉平向量
    eucl_dist_b = eucl_dist_b.reshape(eucl_dist_b.shape[0] * eucl_dist_b.shape[1])
    # 取出非零向量
    # eucl_dist_b = eucl_dist_b[eucl_dist_b != 0]
    print(eucl_dist_b)
    # Calculate score as given in the paper
    dis_mean = torch.mean(eucl_dist_b)
    score = 2 * (1 / (1 + torch.exp(dis_mean)))
    score = score.detach()
    print(score)
    # Normalize value based on alpha and r
    norm_res = 1 / (1 + np.exp(-(0.2 * (score - 0.88))))
    print(norm_res)


def assasadasda():
    import numpy as np

    # 示例数组 a 和 b
    a = np.array([1, 3, 2, 1, 3, 2])
    b = np.array([5, 7, 4, 8, 6, 9])

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

    print("Original Arrays:")
    print("a:", a)
    print("b:", b)
    print("\nResult after Operation:")
    print("Sorted a:", sorted_a)
    print("Sorted b:", sorted_b)


def rrr():
    ss = [0.8300, 0.8467, 0.8433, 0.8533, 0.8200, 0.8533, 0.8567, 0.8333, 0.8400,
          0.8200, 0.8567, 0.8533, 0.8333, 0.8500, 0.8567, 0.8400, 0.8333, 0.8467,
          0.8567, 0.8433, 0.8500, 0.8500, 0.8400, 0.8633]

    sum_number = sum(ss)
    avg = sum_number / len(ss)
    return avg


def sdddssd():
    a = [3, 1, 4, 1, 5, 9, 2, 6, 5]
    b = ['apple', 'banana', 'cherry', 'apple', 'banana', 'cherry', 'apple', 'banana', 'cherry']
    # 对数组a进行排序
    sorted_a = sorted(a)
    # 使用sorted()函数和lambda表达式作为排序键
    sorted_a_b = sorted(zip(a, b), key=lambda x: x[0])
    # 分离排序后的a和b
    sorted_a, sorted_b = zip(*sorted_a_b)
    print("排序后的数组a:", sorted_a)
    print("跟随a变动的数组b:", sorted_b)


def ggggg():
    # import random
    # random.seed(5)
    # index = random.randint(0, 299)
    # ran = random.Random()
    # index2 = ran.randint(0, 299)
    # index3 = ran.randint(0, 299)
    # print(index3)
    # print(index2)
    import torch
    from torch import nn
    lstm = nn.LSTM(768, 768)
    a = torch.randn(100, 3, 768)
    out, (ht,ct) = lstm(a)
    print(out.shape)


def kkkkk():
    commit_list_all = []
    with open("./exp1/video_list.txt", 'r', encoding='utf-8') as fr:
        content = fr.readline()
        while content:
            array = content.split("/sliver")[0]
            commit_list_all.append(array)
            content = fr.readline()
    # F:\multi_view_dataset\\id_user\\week_i\\commit_id\\video1-2-3
    # print(len(commit_list_all))
    commit_res = commit_list_all[:-24]
    commit_list_all23 = fetch_video_list(root=root)


def fetch_video_list(self, root):
    import os
    commit_list_all = []
    # F:\multi_view_dataset\\id_user\\week_i\\commit_id\\video1-2-3
    user_list = [os.path.join(root, user) for user in os.listdir(root)]
    for user in user_list:
        week_user_list = [os.path.join(user, week) for week in os.listdir(user)]
        for week_id in week_user_list:
            commit_list = os.listdir(week_id)
            for commit in commit_list:
                commit_id = os.path.join(week_id, commit)
                commit_list_all.append(commit_id)
    commit_sort = sorted(commit_list_all, key=lambda x: int(x.split("/")[-1]))
    # print(len(commit_sort))
    return commit_sort


if __name__ == "__main__":
    # model = KFS_Environment(total_frame_size=15, initial_state=0)
    # ser_fiq_gpu()
    # norma()
    # import numpy as np
    # observation_space = gym.spaces.MultiDiscrete([[2] * 15] * 3)
    # action_low = np.array([[-1] * 32] * 2)
    # action_high = np.array([[1] * 32] * 2)
    # action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=int)
    # print(action_space.shape)
    # video_path = "/home/HDD/junruit/multi_view_dataset/12345678/week1/99999999/sliver1111.mp4"
    # import os
    # store_path = '/home/HDD/junruit/MVD_Unframe' + video_path.split("multi_view_dataset")[-1] + '/'
    # img_path = "/home/HDD/junruit/multi_view_dataset/12345678/week1/99999999/sliver1111/111.jpg"
    # directory_path = os.path.dirname(img_path)
    # detect_out_path = "/home/HDD/junruit/MV_Face"
    # img_out_path = detect_out_path + img_path.split("multi_view_dataset")[-1]
    # print(img_out_path)
    # from mtcnn import MTCNN
    # import cv2
    # detect_face = MTCNN()
    # img = "8512.jpg"
    # rotate_image(img)
    # # image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    # # result = detect_face.detect_faces(image)
    # # rgb_image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    # # bounding_box = result[0]['box']
    # # bounding_box = [0 if x < 0 else x for x in bounding_box]
    # # print(result)
    # # if len(result) > 0:
    # #     cropped = []
    # #     cropped.append(rgb_image[bounding_box[1]:bounding_box[1] + bounding_box[3],
    # #                    bounding_box[0]:bounding_box[0] + bounding_box[2]])
    # #
    # # print(cropped)
    # # print(len(cropped))
    # # print(len(cropped[0]))
    # import cv2
    #
    # # 加载人脸检测器
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    #
    # # 读取图像
    # image = cv2.imread('8512.jpg')
    #
    # # 转换为灰度图像
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    # # 在灰度图像中检测人脸
    # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    #
    # print(len(faces))
    # # 遍历检测到的人脸并截取出来并保存
    # for i, (x, y, w, h) in enumerate(faces):
    #     face_roi = image[y:y + h, x:x + w]
    #     cv2.imshow(f'Face {i + 1}', face_roi)
    #
    #     # 保存截取的人脸图像
    #     cv2.imwrite(f'face_{i + 1}.jpg', face_roi)
    #
    # # 关闭窗口
    # cv2.waitKey(100)
    # cv2.destroyAllWindows()
    # cal_ten()
    # asasaas()
    # assasadasda()
    # print(rrr())
    # sdddssd()
    ggggg()
