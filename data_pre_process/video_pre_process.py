# -*- encoding: utf-8 -*-
"""
@File    : video_pre_process.py
@Time    : 2023/10/17 14:24
@Author  : junruitian
@Software: PyCharm
"""
import os
import shutil
import cv2
from mtcnn import MTCNN


def get_frame_from_video(video_path, expect_num, interval=None):
    """
   video_name:输入视频路径（除了视频名称之外，读入视频的绝对路径中文件夹一定不要出现中文字符，不然不能保存图片）
   interval: 保存图片的帧率间隔
    """

    video_capture = cv2.VideoCapture(video_path)

    frame_nums = video_capture.get(7)  # 获取视频总帧数
    # print("视频的总帧数为：", int(frame_nums))
    frame_rate = video_capture.get(5)  # 获取视频帧率
    # print("视频的帧率为：", int(frame_rete))
    print(frame_nums)
    i = 0  # i 从 0 开始计数的帧数
    j = 0  # j 从 1 开始，记录第几次间隔
    interval = int(frame_nums / expect_num)
    if interval == 0:
        print("frames too short")
        with open("frame_special.txt", 'a+', encoding='utf-8') as fw:
            fw.write(video_path)
            fw.write(" {}  frames too short".format(frame_nums))
            fw.write("\n")
        interval = 1
    # print(video_path)
    # 创建存帧文件夹

    # video_path "/home/HDD/junruit/multi_view_dataset/12345678/week1/99999999/sliver1111.mp4"
    store_path = '/home/HDD/junruit/MVD_Unframe' + video_path.split("multi_view_dataset")[-1].split(".")[-2] + '/'
    # pwd = os.getcwd() + '\\frame\\' + "\\video1\\"
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    while True:
        # success -boolean 正确读取True 结尾False
        success, frame = video_capture.read()  # 一直在读入视频画面帧

        if not success:
            print('%s frames saved from the video' % j)
            break

        # 判断帧率间隔保存帧
        if i % interval == 0:
            j += 1
            save_name = store_path + str(i) + '.jpg'
            cv2.imwrite(save_name, frame)
            # print('image of %s is saved' % (save_name))

        i += 1
        '''
        i递增置于循环内：防止出现 %同余后最后一帧为空指针的情况 
        [ error: (-215:Assertion failed) !_img.empty() in function ‘cv::imwrite’]
        @solve: 21/12/2 
        '''

    # 传输记录当前视频解帧图片数量
    # frame_num 为总帧数 j 为筛选的帧数
    return frame_nums, j, store_path


def user_unframe(detect, user_list, dataset_path, expect_num):
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
                        if IsSuccess(video_path):
                            continue
                        else:
                            Sign = face_align(detect, video_path, expect_num)
                            if Sign:
                                with open("Video_unframe_success.txt", 'a+', encoding='utf-8') as fw:
                                    fw.write(video_path)
                                    fw.write("\n")

    return True


def IsSuccess(video_path):
    video = video_path.split(".")[0].split("/")[-1]
    with open("Video_unframe_success.txt", 'r', encoding='utf-8') as fr:
        content = fr.read()

    with open("not_enough.txt", 'r', encoding='utf-8') as fr2:
        content2 = fr2.read()

    # content = content + content2
    
    if video in content:
        return True
    else:
        return False


def face_align(detect, video_path, expect_num):
    temp = expect_num
    select_nums = 0
    while True:
        select_num_last_round = select_nums
        video_capture = cv2.VideoCapture(video_path)
        frame_nums = int(video_capture.get(7))  # 获取视频总帧数
        if int(frame_nums / temp) > 0:
            interval = int(frame_nums / temp)
        else:
            interval = 1
        select_nums = 0
        for i in range(0, frame_nums):
            if i % interval == 0:
                select_nums += 1
        print("Select: " + str(select_nums) + "; Last:" + str(select_num_last_round))
        if select_nums != select_num_last_round:
            # beginning delete
            store_path = '/home/HDD/junruit/MVD_Unframe' + video_path.split("multi_view_dataset")[-1].split(".")[
                -2] + '/'
            if os.path.exists(store_path):
                shutil.rmtree(store_path)
                directory_path = os.path.dirname("/home/HDD/junruit/MV_Face" + store_path.split("MVD_Unframe")[-1])
                if os.path.exists(directory_path):
                    shutil.rmtree(directory_path)

            frame_nums, j, frames_path = get_frame_from_video(video_path, temp)

            print(video_path + "," + str(j))
            frame_list = os.listdir(frames_path)
            counter = 0
            for frame in frame_list:
                frame_path = os.path.join(frames_path, frame)
                counter = counter + locate_face(detect, frame_path)
            print("Counter: " + str(counter))
            if counter < 1000:
                if int((1000/counter - 1) * 1000) < 500:
                    temp = temp + 500
                else:
                    temp = temp + int((1000/counter - 1) * 1000)
                # print("Warning with Extra Face detection!")
                # print(video_path)

            else:
                return True
        else:
            if frame_nums == select_nums:
                print("Warning with Short Video Not Enough Faces")
                with open("not_enough.txt", 'a', encoding='utf-8') as fa:
                    fa.write(frames_path)
                    fa.write("\n")
                return False
            else:
                if counter <= 0:
                    counter = 1
                if int((1000 / counter - 1) * 1000) < 500:
                    temp = temp + 500
                else:
                    temp = temp + int((1000 / counter - 1) * 1000)


def rotate_image(image_path, k):
    image = cv2.imread(image_path)
    for i in range(0, k):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


# eg. img_path = "/home/HDD/junruit/MVD_Unframe/12345678/week1/99999999/sliver1111/111.jpg"
def locate_face(detector, img_path):
    detect_out_path = "/home/HDD/junruit/MV_Face"
    img_out_path = detect_out_path + img_path.split("MVD_Unframe")[-1]

    # print(img_out_path)
    img_out_file_directory = os.path.dirname(img_out_path)
    if not os.path.exists(img_out_file_directory):
        os.makedirs(img_out_file_directory)

    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    rgb_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    result = detector.detect_faces(image)

    face_num = len(result)
    # print(result)
    # 检测数组为空 即未检测到人脸情况
    if face_num == 0:
        for round in range(1, 4):
            image = cv2.cvtColor(rotate_image(img_path, round), cv2.COLOR_BGR2RGB)
            rgb_image = rotate_image(img_path, round)
            result = detector.detect_faces(image)
            if len(result) > 0:
                break
        if len(result) == 0:
            # Todo 对于图片中未检测出的特例如何处理 保留or删除
            with open("waiting_delete_path.txt", 'a', encoding='utf-8') as fd:
                fd.write(img_path)
                fd.write("\n")
            # 直接保留源文件
            # print(img_out_path_small)
            # TODO 无脸不应写入，使用时从unframe中取出
            # cv2.imwrite(img_out_path, rgb_image)
            return 0

    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
    bounding_box = result[0]['box']
    keypoints = result[0]['keypoints']

    # 修改一些检测bounding_box出现负值情况 即脸部遮挡等
    bounding_box = [0 if x < 0 else x for x in bounding_box]
    '''
    截取人脸部分并保存为对应图片目录
    '''
    if len(result) > 0:
        cropped = []
        cropped.append(rgb_image[bounding_box[1]:bounding_box[1] + bounding_box[3],
                       bounding_box[0]:bounding_box[0] + bounding_box[2]])

    cv2.imwrite(img_out_path, cropped[0])
    '''
    截图部分end
    '''
    return 1

    # cv2.rectangle(image,
    #               (bounding_box[0], bounding_box[1]),
    #               (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
    #               (0, 155, 255),
    #               2)
    #
    # cv2.circle(image, (keypoints['left_eye']), 2, (0, 155, 255), 2)
    # cv2.circle(image, (keypoints['right_eye']), 2, (0, 155, 255), 2)
    # cv2.circle(image, (keypoints['nose']), 2, (0, 155, 255), 2)
    # cv2.circle(image, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
    # cv2.circle(image, (keypoints['mouth_right']), 2, (0, 155, 255), 2)
    #
    # # img_out_path = "./frame_face_copy/happy_0_0.avi/0.jpg"
    # img_out_path = "./frame_face_copy/" + avi_path + "/" + image_name
    # build_path_need = "./frame_face_copy/" + avi_path  # 文件保存路径，如果不存在就会被重建
    # if not os.path.exists(build_path_need):  # 如果路径不存在
    #     os.makedirs(build_path_need)
    # # print(img_out_path)
    # cv2.imwrite(img_out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # print(result)


if __name__ == '__main__':
    expect_num = 1000
    dataset_path = "/home/HDD/junruit/multi_view_dataset"
    user_list = os.listdir(dataset_path)
    detect_face = MTCNN()
    Bool = user_unframe(detect_face, user_list, dataset_path, expect_num)
