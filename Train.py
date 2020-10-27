import numpy
# import tensorflow
import sys
import os
# sys.path.append("{}".format(os.getcwd()))
# from predict_data_make.py import *
import numpy as np
import scipy.io as scio
import pandas as pd


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
        if time[frames]!=[]:
            radar_time.append(time[frames][0, 3] * 3600 + time[frames][0, 4] * 60 + time[frames][0, 5])
        else:
            point_cloud = np.delete(point_cloud,[frames,])
            length_frames-=1

    for labels_frames in range(length_labels_frames):
        str_kinect_time = labels_csv[0].values[labels_frames]
        temp = str_kinect_time.split(" ")[1].split(":")
        labels_time.append(float(temp[0]) * 3600 + float(temp[1]) * 60 + float(temp[2]) + float("0." + temp[3]))

    for frames in range(length_frames):
        index = np.argmin(np.abs((labels_time - radar_time[frames])), axis=0)
        point_cloud_label.append(labels_csv[76].values[index])

    print("PointCloudshape:{},labelshape:{}".format(np.shape(point_cloud), np.shape(point_cloud_label)))

    return point_cloud, point_cloud_label


def saving(train_data, train_label,point_cloud, label):
    train_data = np.hstack((train_data, point_cloud))
    train_label = np.hstack((train_label,label))
    return train_data,train_label

def debuging():
    data_path = "C:\\Users\\70639wimoc\\PycharmProjects\\RTFALLdetect\\falldata\\move\\yaomove"
    radar_data, labels_csv = load_data(data_path)
    point_cloud, point_cloud_label = labing(radar_data, labels_csv)
def main():
    train_data = np.array([])
    train_label = np.array([])
    falldata_path ="C:\\Users\\70639wimoc\\PycharmProjects\\RTFALLdetect\\falldata\\"
    falldata_folder_list = os.listdir(falldata_path)

    for ff_name in falldata_folder_list:
        target_path = falldata_path + ff_name
        folder_list = os.listdir(target_path)
        for folder_name in folder_list:

            data_path = "{}\\{}".format(target_path, folder_name)
            print("processing {}...".format(folder_name))
            radar_data, labels_csv = load_data(data_path)
            point_cloud, point_cloud_label = labing(radar_data, labels_csv)
            train_data,train_label = saving(train_data, train_label,point_cloud, point_cloud_label)
            print("Data_shape:{}//{}".format(np.shape(train_data),np.shape(train_label)))



if __name__ == "__main__":
    main()
