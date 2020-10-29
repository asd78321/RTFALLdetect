import numpy
# import tensorflow
import sys
import os
# sys.path.append("{}".format(os.getcwd()))
# from predict_data_make.py import *
import numpy as np
import scipy.io as scio
import pandas as pd
import matplotlib.pyplot as plt


def load_data(data_path):
    radar_data_name = '\\fhistRT.mat'
    kinect_data_name = '\\labels.csv'

    file_key_name = 'fHist'
    radar_data = scio.loadmat(data_path + radar_data_name)[file_key_name]
    labels_csv = pd.read_csv(data_path + kinect_data_name, names=range(77))

    return radar_data, labels_csv


def labing(radar_data, labels_csv):
    radar_time = []
    labels_time = []
    point_cloud_label = []
    length_labels_frames = np.shape(labels_csv)[0]
    length_frames = np.shape(radar_data)[1]
    point_cloud = np.reshape(radar_data['pointCloud'].T, [length_frames, ])  # transpose matrix
    time = radar_data['TIME'][0].T

    for frames in range(length_frames):
        if np.size(time[frames]) != 0:
            radar_time.append(time[frames][0, 3] * 3600 + time[frames][0, 4] * 60 + time[frames][0, 5])
        else:
            point_cloud = np.delete(point_cloud, [frames, ])
            length_frames -= 1

    for labels_frames in range(length_labels_frames):
        str_kinect_time = labels_csv[0].values[labels_frames]
        temp = str_kinect_time.split(" ")[1].split(":")
        labels_time.append(float(temp[0]) * 3600 + float(temp[1]) * 60 + float(temp[2]) + float("0." + temp[3]))

    for frames in range(length_frames):
        index = np.argmin(np.abs((labels_time - radar_time[frames])), axis=0)
        point_cloud_label.append(labels_csv[76].values[index])

    print("PointCloudshape:{},labelshape:{}".format(np.shape(point_cloud), np.shape(point_cloud_label)))

    return point_cloud, point_cloud_label


def saving(train_data, train_label, point_cloud, label):
    train_data = np.hstack((train_data, point_cloud))
    train_label = np.hstack((train_label, label))
    return train_data, train_label


def debuging():
    data_path = "C:\\Users\\70639wimoc\\PycharmProjects\\RTFALLdetect\\falldata\\move\\yaomove"
    radar_data, labels_csv = load_data(data_path)
    point_cloud, point_cloud_label = labing(radar_data, labels_csv)


def main_data_processing():
    saving_name = "./raw_data"
    train_data = np.array([])
    train_label = np.array([])
    falldata_path = "C:\\Users\\70639wimoc\\PycharmProjects\\RTFALLdetect\\falldata\\"
    falldata_folder_list = os.listdir(falldata_path)

    for ff_name in falldata_folder_list:
        target_path = falldata_path + ff_name
        folder_list = os.listdir(target_path)
        for folder_name in folder_list:
            data_path = "{}\\{}".format(target_path, folder_name)
            print("processing {}...".format(folder_name))
            radar_data, labels_csv = load_data(data_path)
            point_cloud, point_cloud_label = labing(radar_data, labels_csv)
            train_data, train_label = saving(train_data, train_label, point_cloud, point_cloud_label)
            print("Data_shape:{}//{}".format(np.shape(train_data), np.shape(train_label)))

    train_data, train_label = check_null_data(train_data, train_label)

    np.save(saving_name, train_data)
    np.save("{}_label".format(saving_name), train_label)


def check_null_data(train_data, train_label):
    null_index_list = []
    Ptrain_data = train_data
    Ptrain_label = []
    print("Raw train_data length Data:{}//Label:{}".format(len(train_data), len(train_label)))
    null_count = 0
    for frames in range(len(train_data)):
        if np.size(train_data[frames]) == 0:
            null_index_list.append(frames)
            null_count += 1
        else:
            Ptrain_data[frames - null_count] = train_data[frames]
            Ptrain_label.append(train_label[frames])

    print("empty array length:{}".format(len(null_index_list)))

    train_label = np.array(Ptrain_label)
    train_data = Ptrain_data[0:len(train_data) - len(null_index_list)]
    print("train_data Shape Data:{}//Label:{}".format(np.shape(train_data), np.shape(train_label)))

    return train_data, train_label


def load_train_data_processing(data_path):
    data_name = "raw_data"
    train_data = np.load("{}.npy".format(data_path + data_name), allow_pickle=True)
    train_label = np.load("{}_label.npy".format(data_path + data_name))
    print("loaded training data....shape is {}".format(np.shape(train_data)))

    return train_data, train_label


def testing(train_data, train_label):
    # Total data = 88659
    # 0 = 站或移動 :8956
    # 1 = 站到坐   :15978
    # 2 = 坐到站   :14927
    # 3 = 坐到躺   :11604
    # 4 = 躺到坐   :11899
    # 5 = 跌倒     :11680
    # 6 = 起身     :13615
    length_image = 89

    target_index_list = np.where(train_label == 5)
    print("Pose length: {}".format(len(target_index_list[0])))
    image_data = target_index_list[0][0:length_image]
    image = []
    null_count = 0
    for i in image_data:
        image.append(np.mean(train_data[i][3]))  # Doppler index = 3
    plot_image(image)


def testing_2(train_label):
    pose_list = ["stand or walk", "stand to sit", "sit to stand", "sit to lie", "lie to sit", "fall", "get up"]

    for pose_index in range(len(pose_list)):
        target_index_list = np.where(train_label == pose_index)
        # print(target_index_list)
        len_list = []
        len_index = []
        len_count = 0
        for i in range(len(target_index_list[0])):
            if i == 0:
                temp = target_index_list[0][i]
            else:
                now = target_index_list[0][i]
                if now - temp < 5:
                    len_count += 1
                    temp = target_index_list[0][i]
                else:
                    if len_count > 10:
                        len_list.append(len_count)
                        len_index.append(i)
                        len_count = 0
                        temp = target_index_list[0][i]

        min_index = np.where(len_list == np.min(len_list))
        max_index = np.where(len_list == np.max(len_list))
        # min
        min_star_index = len_index[min_index[0][0]] - np.min(len_list)
        min_end_index = len_index[min_index[0][0]]
        image_data = target_index_list[0][min_star_index:min_end_index]
        image = []
        null_count = 0
        for i in image_data:
            image.append(np.mean(train_data[i][3]))  # Doppler index = 3
        for count in range(len_list[max_index[0][0]] - len(image)):
            image.append(0)
        # plot_image(image, pose_list[pose_index] + "min")

        # max
        max_star_index = len_index[max_index[0][0]] - np.max(len_list)
        max_end_index = len_index[max_index[0][0]]
        image_data = target_index_list[0][max_star_index:max_end_index]
        image = []
        null_count = 0
        for i in image_data:
            image.append(np.mean(train_data[i][3]))  # Doppler index = 3
        # plot_image(image, pose_list[pose_index] + "max")
        de_max_min_len_list = np.delete(len_list, [max_index[0][0], min_index[0][0]])
        # print("length_list:{}".format(len_list))
        comform_index = np.where(np.array(len_list) < round(np.mean(de_max_min_len_list)+20))

        print(
            "Pose: {} //Count: {}組 De_Count: {}組//Meanlength: {:.2f}秒 \n //de_Max_meanLength: {:.2f}秒//原始最長: {:.2f}秒//原始最短: {:.2f}秒//標準差:{:.2f} //去對最大標準差:{:.2f} ".format(
                pose_list[pose_index], len(len_list), len(comform_index[0]),
                np.mean(len_list) / 10, np.mean(de_max_min_len_list) / 10,
                np.max(len_list) / 10, np.min(len_list) / 10, np.std(len_list), np.std(de_max_min_len_list)))


def plot_image(image, pose):
    plt.title("{}".format(pose))
    plt.xlabel("Time")
    plt.ylabel("Doppler")
    plt.plot(image)
    plt.show()


if __name__ == "__main__":
    # main_data_processing()
    data_path = "C:\\Users\\70639wimoc\\PycharmProjects\\RTFALLdetect\\"
    train_data, train_label = load_train_data_processing(data_path)
    # testing(train_data, train_label)
    testing_2(train_label)
