# import tensorflow
import sys
import os
import numpy as np
import scipy.io as scio
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import xlwt, xlrd
from mpl_toolkits.mplot3d import Axes3D


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
            train_data, train_label = stack_and_saving(train_data, train_label, point_cloud, point_cloud_label)
            print("Data_shape:{}//{}".format(np.shape(train_data), np.shape(train_label)))

    train_data, train_label = check_null_data(train_data, train_label)

    np.save(saving_name, train_data)
    np.save("{}_label".format(saving_name), train_label)


def load_data(data_path):
    print("load data....")
    radar_data_name = '\\fhistRT2.mat'
    kinect_data_name = '\\labels.csv'

    file_key_name = 'fHist'
    radar_data = scio.loadmat(data_path + radar_data_name)[file_key_name]
    labels_csv = pd.read_csv(data_path + kinect_data_name, names=range(77))

    return radar_data, labels_csv


def labing(radar_data, labels_csv):
    print("labing....")
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


def stack_and_saving(train_data, train_label, point_cloud, label):
    print("stacking....")
    train_data = np.hstack((train_data, point_cloud))
    train_label = np.hstack((train_label, label))
    return train_data, train_label


def check_null_data(train_data, train_label):
    print("checking null data...")
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


def split_and_resize(train_data, train_label):  # Doppler
    pose_list = ["stand_or_walk", "stand_to_sit", "sit_to_stand", "sit_to_lie", "lie_to_sit", "fall", "get_up"]

    for pose_index in range(len(pose_list)):
        target_index_list = np.where(train_label == pose_index)
        print("processing {}...".format(pose_list[pose_index]))
        len_list = []
        len_index = []
        len_count = 0
        for i in range(len(target_index_list[0])):
            if i == 0:
                temp = target_index_list[0][i]  # 上一幀
            else:
                now = target_index_list[0][i]  # 當前幀
                if now - temp < 2:
                    len_count += 1
                    temp = target_index_list[0][i]
                else:
                    if len_count > 10:  # 連續動作且超過1秒
                        len_list.append(len_count)  # 各別動作長度(幀)
                        len_index.append(i)  # 切換幀之index
                        len_count = 0
                        temp = target_index_list[0][i]

        min_index = np.where(len_list == np.min(len_list))  # 最長\短 幀之index
        max_index = np.where(len_list == np.max(len_list))

        data = []
        for data_count in range(len(len_list)):
            star_index = len_index[data_count] - len_list[data_count]
            end_index = len_index[data_count]
            doopler_data = []
            doppler_data_index = target_index_list[0][star_index:end_index]
            for j in doppler_data_index:
                doopler_data.append(np.mean(train_data[j][3]))

            doopler_data = np.array(doopler_data)
            doopler_data.resize((70, 1))
            # doopler_data.resize((int(round(np.mean(len_list))), 1))
            data.append(doopler_data)

        data = np.array(data)
        if pose_index == 2:
            data = np.delete(data, 194, axis=0)
        np.save("./training_data_doppler//{}".format(pose_list[pose_index]), data)  # sit to stand frames:194 have error

        # de_max_min_len_list = np.delete(len_list, [max_index[0][0], min_index[0][0]])
        # comform_index = np.where(np.array(len_list) < round(np.mean(de_max_min_len_list) + 20))
        #
        # print(
        #     "Pose: {} //Count: {}組 De_Count: {}組//Meanlength: {:.2f}秒 \n //de_Max_meanLength: {:.2f}秒//原始最長: {:.2f}秒//原始最短: {:.2f}秒//標準差:{:.2f} //去對最大標準差:{:.2f} ".format(
        #         pose_list[pose_index], len(len_list), len(comform_index[0]),
        #         np.mean(len_list) / 10, np.mean(de_max_min_len_list) / 10,
        #         np.max(len_list) / 10, np.min(len_list) / 10, np.std(len_list), np.std(de_max_min_len_list)))


def split_and_pointcloud(train_data, train_label):
    pose_list = ["stand_or_walk", "stand_to_sit", "sit_to_stand", "sit_to_lie", "lie_to_sit", "fall", "get_up"]
    for pose_index in range(len(pose_list)):
        target_index_list = np.where(train_label == pose_index)
        print("processing {}...".format(pose_list[pose_index]))
        len_list = []
        len_index = []
        len_count = 0
        print(np.shape(target_index_list[0]))
        for i in range(len(target_index_list[0])):
            if i == 0:
                temp = target_index_list[0][i]  # 上一幀
            else:
                now = target_index_list[0][i]  # 當前幀
                if now - temp < 2:
                    len_count += 1
                    temp = target_index_list[0][i]
                else:
                    if len_count > 10:  # 連續動作且超過1秒
                        len_list.append(len_count)  # 各別動作長度(幀)
                        len_index.append(i)  # 切換幀之index
                        len_count = 0
                        temp = target_index_list[0][i]
        #
        min_index = np.where(len_list == np.min(len_list))  # 最長\短 幀之index
        max_index = np.where(len_list == np.max(len_list))
        #
        f_pointcloud = []
        cen_Point_cloud = []
        print("dataLength:{}".format(len(len_list)))
        for data_count in range(len(len_list)):
            star_index = len_index[data_count] - len_list[data_count]
            end_index = len_index[data_count]
            frame_Point_cloud = []
            Point_cloud_index = target_index_list[0][star_index:end_index]
            for j in Point_cloud_index:
                frame_Point_cloud.append(np.array(train_data[j]))

            centroid, pointcloud = pointCloud_resize(frame_Point_cloud)

            # centroid.resize((int(round(np.mean(len_list))), 3))
            # pointcloud.resize((int(round(np.mean(len_list))), 100, 3))
            centroid.resize((70, 3))
            pointcloud.resize((70, 100, 3))
            cen_Point_cloud.append(centroid)
            f_pointcloud.append(pointcloud)

        cen_Point_cloud = np.array(cen_Point_cloud)
        f_pointcloud = np.array(f_pointcloud)

        print("cen:{} po:{}".format(np.shape(cen_Point_cloud), np.shape(f_pointcloud)))
        if pose_index == 2:
            cen_Point_cloud = np.delete(cen_Point_cloud, 194, axis=0)
            f_pointcloud = np.delete(f_pointcloud, 194, axis=0)
        np.save("./training_data_Height//{}".format(pose_list[pose_index]),cen_Point_cloud)
        np.save("./training_data_point2d//{}".format(pose_list[pose_index]), f_pointcloud)
        # print(np.shape(Point_cloud))

    # sit to stand frames:194 have error


def pointCloud_resize(frames_point_cloud):
    # range:0 ,theta:1 ,phi:2 ,doppler:3 ,snr:4
    pl_f = []
    frames_centroid_point = np.zeros([len(frames_point_cloud), 3])
    sample_frames_point_cloud=np.zeros([len(frames_point_cloud),100, 3])
    for point_count in range(len(frames_point_cloud)):
        RANGE = frames_point_cloud[point_count][0]
        Theta = frames_point_cloud[point_count][1]
        Phi = frames_point_cloud[point_count][2]
        N = 100
        centroid_point, point_cloud = oversample(RANGE, Theta, Phi, N)
        frames_centroid_point[point_count] = centroid_point
        sample_frames_point_cloud[point_count] =point_cloud
    return frames_centroid_point, sample_frames_point_cloud


def oversample(RANGE, Theta, Phi, N):
    centroid_point = np.zeros([3])
    point_cloud = np.zeros([len(RANGE), 3])
    point_cloud[:, 0] = RANGE * np.cos(Phi) * np.sin(Theta)
    point_cloud[:, 1] = RANGE * np.cos(Phi) * np.cos(Theta)
    point_cloud[:, 2] = RANGE * np.sin(Phi)

    centroid_point[0] = np.mean(point_cloud[:, 0])  # centroid x
    centroid_point[1] = np.mean(point_cloud[:, 1])  # centroid y
    centroid_point[2] = np.mean(point_cloud[:, 2])  # centroid z

    new_point_cloud = np.zeros([N, 3])

    S = np.sqrt(N / len(point_cloud))
    for i in range(N):
        if i < len(point_cloud):
            new_point_cloud[i, 0] = S * point_cloud[i, 0] + centroid_point[0] - S * centroid_point[0]
            new_point_cloud[i, 1] = S * point_cloud[i, 1] + centroid_point[1] - S * centroid_point[1]
            new_point_cloud[i, 2] = S * point_cloud[i, 2] + centroid_point[2] - S * centroid_point[2]
        else:
            new_point_cloud[i, 0] = centroid_point[0]
            new_point_cloud[i, 1] = centroid_point[1]
            new_point_cloud[i, 2] = centroid_point[2]
    # print(np.shape(centroid_point))
    return centroid_point, new_point_cloud


def reading_log_and_plot():
    ylabel = "%"
    xlabel = "epoch"

    with open('trainHistoryDict.txt', 'rb') as file_pi:
        history = pickle.load(file_pi)
        for data_name in history.keys():
            plot_image(history[data_name], data_name, xlabel, ylabel)


def plot_image(image, pose, xlabel, ylabel):
    plt.title("{}".format(pose))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(image)
    plt.show()


def set_pltdata_doppler(data_path):
    pltdata = []
    file_name_list = os.listdir(data_path)
    for file_name_index in range(len(file_name_list)):
        # print(file_name_list[file_name_index])
        data = np.load(data_path + "/{}".format(file_name_list[file_name_index]))
        # data = data.reshape([len(data), 70])
        plt.title('{}'.format(file_name_list[file_name_index]))
        for i in range(len(data)):
            # plt.plot(data[i])
            if i == 0:
                plot_data = data[i]
            else:
                plot_data = plot_data + data[i]
        plot_data = plot_data / len(data)
        pltdata.append(plot_data)
    return pltdata


def set_pltdata_pointcloud(data_path):
    pltdata = []
    file_name_list = os.listdir(data_path)
    # data = np.load(data_path+"/{}".format(file_name_list[0]))
    # z = data[:,:,2]
    for file_name_index in range(len(file_name_list)):
        data = np.load(data_path + "/{}".format(file_name_list[file_name_index]), allow_pickle=True)
        data = data[:, :, 2]  # plot z
        plt.title('{}'.format(file_name_list[file_name_index]))
        for i in range(len(data)):
            # plt.plot(data[i])
            if i == 0:
                plot_data = data[i]
            else:
                plot_data = plot_data + data[i]
        plot_data = plot_data / len(data)
        pltdata.append(plot_data)
    return pltdata


def plot_doppler_cen():
    data_path = "./temp_height"
    test1 = set_pltdata_pointcloud(data_path)
    data_path = "./temp_doppler"
    test2 = set_pltdata_doppler(data_path)

    file_name_list = os.listdir(data_path)
    for i in range(len(file_name_list)):
        p1 = plt.plot(test1[i])
        p2 = plt.plot(test2[i])
        plt.title("{}".format(file_name_list[i]))
        plt.xlabel('Frames')
        plt.legend(["cen_height", "cen_doppler"])
        plt.show()


def plot_pointcloud_scatter():
    data_path = "./training_data_point2d"
    file_list = os.listdir(data_path)
    for file_name in file_list:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        data = np.load(data_path + "/{}".format(file_name))
        # data.reshape([np.shape(data)[0]*np.shape(data)[1],100,3])
        for i in range(np.shape(data)[0]):
            if i == 0:
                plot_data = data[i]
            else:
                plot_data = plot_data + data[i]
        plot_data = plot_data / np.shape(data)[0]
        ax.scatter(data[:,:,0],data[:,:,1])
        # plt.scatter(plot_data[:, :, 0], plot_data[:, :, 1])
        # plt.title("{}".format(file_name))
        plt.show()


def processing_multi_fhistRT(data_path):
    file_key_name = 'fHist'

    matfilenamelist = [os.path.join(data_path, _) for _ in os.listdir(data_path) if _.endswith('.mat')]
    train_data = np.array([])
    for fileindex in matfilenamelist:
        data = scio.loadmat(fileindex)[file_key_name]
        length_frames = np.shape(data)[1]
        point_cloud = np.reshape(data['pointCloud'].T, [length_frames, ])
        if train_data.size > 0:
            train_data = np.concatenate((point_cloud, train_data))
        else:
            train_data = point_cloud
    # train_data = train_data[400:len(train_data) - 200]

    return train_data





def test():
    data_path = "C:\\Users\\70639wimoc\\PycharmProjects\\RTFALLdetect\\walk_stand_xie\\walk_stand_xie"
    data1 = processing_multi_fhistRT(
        data_path="C:\\Users\\70639wimoc\\PycharmProjects\\RTFALLdetect\\walk_stand_xie\\walk_stand_xie")
    data2 = processing_multi_fhistRT(
        data_path="C:\\Users\\70639wimoc\\PycharmProjects\\RTFALLdetect\\walk_stand_feng\\walk_stand_feng")
    raw_data = np.concatenate((data1, data2))
    length = len(raw_data)
    index = []
    for i in range(len(raw_data)):
        if np.size(raw_data[i])==0:
            index.append(i)
    data = np.delete(raw_data,index,axis=0)
    # print(np.shape(raw_data))
    # print(np.shape(data))

    data =data[0:7000]
    split = 70
    train_data_doppler =[]
    train_data_pointcloud =[]
    train_data_height=[]
    for i in range(len(data) // 70):
        doppler_data_list = []
        start_index = i * 70
        end_index = start_index + 70
        frames_data = data[start_index:end_index]
        for j in range(len(frames_data)):
            doppler_data_list.append(np.mean(frames_data[j][3]))
        centroid, pointcloud = pointCloud_resize(frames_data)
        doppler_data_list=np.array(doppler_data_list)
        train_data_doppler.append(doppler_data_list)
        train_data_height.append(centroid)
        train_data_pointcloud.append(pointcloud)

    train_data_doppler =np.array(train_data_doppler)
    train_data_height=np.array(train_data_height)
    train_data_pointcloud =np.array(train_data_pointcloud)

        # plt.plot(doppler_data_list)
        # plt.show()
    print("doopler:{} cen:{} pointclud:{}".format(np.shape(train_data_doppler),np.shape(train_data_height),np.shape(train_data_pointcloud)))
    np.save("./temp/stand_or_walk_p",train_data_pointcloud)
    np.save("./temp/stand_or_walk_d", train_data_doppler)
    np.save("./temp/stand_or_walk_c", train_data_height)


if __name__ == "__main__":
    # stap1: main_data_processing()
    # main_data_processing() # 資料前處理並串接儲存成raw_data

    # step2: data_features_save *.npy
    data_path = "C:\\Users\\70639wimoc\\PycharmProjects\\RTFALLdetect\\"
    train_data, train_label = load_train_data_processing(data_path)  # labeling and load
    split_and_resize(train_data, train_label)  # 切出每個動作並固定維度,各別儲存
    split_and_pointcloud(train_data, train_label)

    # plot_doppler_cen() # 畫Doppler趨勢圖

    # plot_pointcloud_scatter()# 畫點雲圖

    # data analysis
    # reading_log_and_plot() #畫tensor board

    # test()
