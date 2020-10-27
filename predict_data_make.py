#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import heapq
import numpy as np
import pandas as pd
import scipy.io as scio
import os
import glob
import time
import csv


from sklearn.cluster import DBSCAN



def join_data_to_newdata(data):
    a = data["targetFrameNum"].T
    b = data['TIME'].T
    c = data['pointCloud'].T
    newdata = np.hstack((a, b))
    newdata = np.hstack((newdata, c))

    return newdata

def delet_empty(newdata):
    a = []
    for i in range(len(newdata)):
        if newdata[i, 2].size == 0:
            a.append(i)
    newdata = np.delete(newdata, a, 0)
    return newdata

def voxalize(x_points, y_points, z_points, x, y, z, velocity):
    
#     x_min = np.min(x)
#     x_max = np.max(x)

#     y_min = np.min(y)
#     y_max = np.max(y)

#     z_max = np.max(z)
#     z_min = np.min(z)
    
    x_min = -3.0
    x_max = 3.0

    y_min = 0.0
    y_max = 2.5

    z_max = 3.0
    z_min = -3.0


    z_res = (z_max - z_min) / z_points
    y_res = (y_max - y_min) / y_points
    x_res = (x_max - x_min) / x_points
    if z_min == z_max:
        z_res = 1
    if y_min == y_max:
        y_res = 1
    if x_min == x_max:
        x_res = 1

    #     新方法求取矩陣點


    pixel_x_y=np.zeros([x_points * y_points ])
    pixel_y_z=np.zeros([z_points * y_points ])
    pixel_x_z=np.zeros([x_points * z_points ])

    for i in range(len(y)):
        x_pix = (x[i] - x_min) // x_res
        y_pix = (y[i] - y_min) // y_res
        z_pix = (z[i] - z_min) // z_res

        if x_pix == x_points:
            x_pix = x_points - 1
        if y_pix == y_points:
            y_pix = y_points - 1
        if z_pix == z_points:
            z_pix = z_points - 1

        if x[i]>-3.0 and x[i]<3.0 and y[i]<2.5 and y[i]>0 and z[i]>-3.0 and z[i]<3.0:
            pixel_x_y[int((y_pix)*x_points+x_pix)] = pixel_x_y[int((y_pix)*x_points+x_pix)] + 1
            pixel_y_z[int((y_pix)*z_points+z_pix)] = pixel_y_z[int((y_pix)*z_points+z_pix)] + 1
            pixel_x_z[int((z_pix)*x_points+x_pix)] = pixel_x_z[int((z_pix)*x_points+x_pix)] + 1
 
        else:
            print("！！！！！！！！！！！！！！！有點超出voxesl範圍！"+"\n"+"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    return   pixel_x_y,pixel_y_z,pixel_x_z

sub_dirs = ["0", "1", "2", "0_1", "1_0", "0_2", "2_0", "1_2", "2_1"]

def one_hot_encoding(y_data, sub_dirs, categories):
    Mapping = dict()

    count = 0
    #     將['boxing','jack','jump','squats','walk']編碼
    for i in sub_dirs:
        Mapping[i] = count
        count = count + 1

    y_features2 = []
    for i in range(len(y_data)):
        Type = y_data[i]
        lab = Mapping[Type]
        y_features2.append(lab)

    y_features = np.array(y_features2)
    y_features = y_features.reshape(y_features.shape[0], 1)
    from keras.utils import to_categorical
    y_features = to_categorical(y_features)

    return y_features




def contain_all_files_data_labels(path):
    file_path_first = path
    file_list_first = os.listdir(file_path_first)
    print("要合併的文件夾有：", file_list_first)
    # 姚這層
    data_first_list = []
    labels_list_first = []
    # 遍歷跑程序的多人資料文件夾下所有文件並且合併
    for j in range(len(file_list_first)):
        #     yao站坐倒這層
        file_path_second = file_path_first + file_list_first[j] + "/"
        file_list_second = os.listdir(file_path_second)
        print("file_list_second的路徑：", file_path_second)

        data_list_second = []
        #    遍歷姚下所有文件並且合併
        print("file_list_second中有幾個文件:", len(file_list_second))
        labels_list_second = []
        for k in range(len(file_list_second)):
            file_path_third = file_path_second + file_list_second[k] + "/"
            file_list_third = os.listdir(file_path_third)
            file_mat_list = []

            for i in file_list_third:  # 循环读取路径下的文件并筛选mat文件

                if os.path.splitext(i)[1] == ".mat":  # 筛选mat文件
                    file_mat_list.append(i)

            data_list_third = []
            #     遍歷姚站座談下所有文件合併
            for i in range(0, len(file_mat_list)):
                data_mat_File = file_path_third + file_mat_list[i]
                print("正在讀取這個mat檔案:", data_mat_File)
                temp = scio.loadmat(data_mat_File)['fHist']
                temp = np.delete(temp, [0], axis=1)

                newdata = join_data_to_newdata(temp)
                newdata = delet_empty(newdata)
                print("newdata.shape:", newdata.shape)
                data_list_third.append(newdata)
                del newdata
            data = np.concatenate(data_list_third, axis=0)
            del data_list_third
            print(file_path_third + "此文件大小:", data.shape)
            data_third = data[data[:, 0].argsort()]
            del data
            data_list_second.append(data_third)
            del data_third

            #         加入含有骨架點的label
            kinect_data = pd.read_csv(file_path_third + "labels.csv", names=range(77))
            # 放置转换的时间

            kinect_data[77] = 0

            # 增加時間序列
            str_totime = kinect_data[0].values
            str_totime.shape
            for i in range(len(str_totime)):
                #     將每一個時間string拆分成['21', '42', '25', '836']
                temp = str_totime[i].split(" ")[1].split(":")
                time = float(temp[0]) * 3600 + float(temp[1]) * 60 + float(temp[2]) + float("0." + temp[3])
                kinect_data.iloc[i, 77] = time

            # 去除缺失值
            kinect_data = kinect_data.dropna()

            kinect_data[kinect_data[2].isin([1.1])]

            # 去除kinec中坏的数据
            kinect_data = kinect_data[~kinect_data[2].isin([1.1])]

            kinect_data = kinect_data.reset_index(drop=True)
            labels_list_second.append(kinect_data)
        labels_76_second = np.concatenate(labels_list_second, axis=0)
        labels_list_first.append(labels_76_second)
        print(file_path_second + "此文件labels大小:", labels_76_second.shape)

        #     data_second 中有姚文件夾中所有文件
        data_second = np.concatenate(data_list_second, axis=0)
        print(file_path_second + "此文件大小:", data_second.shape)
        data_first_list.append(data_second)
    labels_76_first = np.concatenate(labels_list_first, axis=0)
    del labels_list_first
    data_first = np.concatenate(data_first_list, axis=0)
    del data_first_list
    data_final = data_first
    del data_first
    labels_76_final = labels_76_first
    print("最終合併文件大小：", data_final.shape)
    print("最終LABLES_76大小：", labels_76_final.shape)
    mat_data = pd.DataFrame(data_final, columns=("targect", "time", "pointCloud"))

    a = []
    for i in range(len(data_final)):
        aa = data_final[:, 1].reshape((-1, 1))
        aa = aa[i, 0]
        time = aa[0, 3] * 3600 + aa[0, 4] * 60 + aa[0, 5]
        a.append(time)
    mat_data["time_s"] = a
    del data_final

    kinect_data = pd.DataFrame(labels_76_final)

    # 将kinect中所有与poincloud中时间最近的数据找出，存入M中
    a = kinect_data[77].values
    m = []

    times = []
    for i in range(len(mat_data)):
        temp = np.abs(mat_data["time_s"][i] - a)
        temp_index = np.argmin(temp, axis=0)
        b = kinect_data.iloc[temp_index, 1:77]

        #     t为kinect的索引
        t = kinect_data.iloc[temp_index, 77]
        m.append(b)

        times.append(t)

    tt = mat_data["time_s"].values.reshape(-1, 1)
    t = np.array(times).reshape(-1, 1)
    print(tt.shape)
    print(t.shape)

    timess = np.c_[tt, t]

    diff = abs(timess[:, 0] - timess[:, 1])

    erro = np.where(diff > 0.1)
    erro = np.squeeze(erro)
    print("erro的長度：",erro.shape)
    print("刪除erro前的數據集：",mat_data.shape)
    # 删除时间差大于0.1秒的数据
    mat_data_temp_final = mat_data.drop(index=erro)
    print("刪除后的數據集大小：",mat_data_temp_final.shape)
    M = np.array(m)

    print(M.shape)

    # 删除时间差大于0.1秒的数据
    kinect_data_final = np.delete(M, erro, axis=0)

    kinect_76_csv = pd.DataFrame(kinect_data_final, columns=range(76))
    #     kinect_75_csv["gesture"]=labels_9kinds_final
    # 将第一列由于NOhuman造成的string类型转换为float类型
    kinect_76_csv[0] = kinect_76_csv[0].apply(lambda x: float(x))
    print("最終合成data長度：", mat_data_temp_final.shape)
    print("最終合成lables長度：", kinect_76_csv.shape)
    kinect_76_csv = kinect_76_csv.reset_index(drop=True)
    # kinect_76_csv.to_csv("./11.28骨架点和labels.csv", index=False, header=False)
    del kinect_data_final
    del M
    del diff
    del kinect_data
    del mat_data
    
    return mat_data_temp_final, kinect_76_csv



















def data_to_picture(mat_data_temp_final, kinect_75_labels, x_cut, y_cut, z_cut, together, slide):
    # 將pointcloud數據轉換成x,y,z
    data = dict()
    for i in range(len(mat_data_temp_final["pointCloud"])):
        data[i] = []
        x_y_z = mat_data_temp_final["pointCloud"].values[i][:4, :].reshape(4, -1)
        for j in range(x_y_z.shape[1]):
            temp = x_y_z[:, j]
            R = temp[0] * np.cos(temp[2])
            z = temp[0] * np.sin(temp[2])
            y = 2.9 - R * np.cos(temp[1])
            x = R * np.sin(temp[1])
            if x < 2 and x > -2 and z > -2 and z < 2 and y < 2.5 and y > 0:
                vel_c = temp[3]
                point_x_y_z_vel_c = [x, y, z, vel_c]
                data[i].append(point_x_y_z_vel_c)


    data_pro = data
    del data
    data_pro1 = dict()
    valid = 0
    del_list = []
    all_noise_index = []
    for i in range(len(data_pro)):
        if data_pro[i] == []:
            del_list.append(i)
        if data_pro[i] != []:
            data1 = np.array(data_pro[i]) * [1, 0.5, 1, 1]
            #                 print(data1.shape)
            estimator = DBSCAN(eps=0.5, min_samples=8, metric='euclidean')
            # 聚类数据
            estimator.fit(data1[:, :3])
            point_labels = estimator.labels_
            #                 print("所有的點的好壞標簽：",set(point_labels))
            if np.unique(point_labels).tolist() == [-1]:
                #                     print("我想看的：",np.unique(point_labels).tolist())
                #                     print(noise_labels)
                all_noise_index.append(i)
                del_list.append(i)
            #                 print("all_noise_index长度：",len(all_noise_index))
            else:
                index_erro_point = np.squeeze(np.where(np.array(point_labels) == -1)).tolist()
                #                     print("index_erro_point的長度:",index_erro_point)
                #                     print("刪除前一個矩陣的長度：",len(data1))
                data_frame = np.delete(data1, index_erro_point, axis=0)
                data_pro1[valid] = data_frame / [1, 0.5, 1, 1]
                #                     print("刪除后矩陣的長度：",len(data_frame))
                valid += 1

    #         print("全部是noise的點雲數量：",len(all_noise_index))
    kinect_75_labels = kinect_75_labels.drop(index=del_list)
    kinect_75_labels = kinect_75_labels.reset_index(drop=True)
    print("kinect_75_labels的shape:", kinect_75_labels.shape)

    print("datapro1.shape:", len(data_pro1))


    labels = kinect_75_labels[75]
    print("labels.unique:", np.unique(labels))
    print("labels.len:", labels.shape)
    pixels1 = []
    pixels2 = []
    pixels3 = []
    for i in data_pro1:
        f = data_pro1[i]
        f = np.array(f)

        # y and z points in this cluster of frames
        x_c = f[:, 0]
        y_c = f[:, 1]
        z_c = f[:, 2]
        vel_c = f[:, 3]

        pix0 = voxalize(x_cut, y_cut, z_cut, x_c, y_c, z_c, vel_c)[0]
        #             pix1為一般模型的voxels
        pix1 = voxalize(x_cut, y_cut, z_cut, x_c, y_c, z_c, vel_c)[1]
        #           pix2為vlstm模型的voxels
        pix2 = voxalize(x_cut, y_cut, z_cut, x_c, y_c, z_c, vel_c)[2]
        #             pixels1為一般模型的voxels
        pixels1.append(pix0)
        pixels2.append(pix1)
        pixels3.append(pix2)
        del pix1
        del pix2
        del pix0

    pixels1 = np.array(pixels1)
    pixels2 = np.array(pixels2)
    pixels3 = np.array(pixels3)

    frames_together = together
    sliding = slide

    train_data1 = []
    train_data2 = []
    train_data3 = []

    i = 0
    while i + frames_together <= pixels1.shape[0]:
        local_data1 = []
        local_data2 = []
        local_data3 = []
        for j in range(frames_together):
            local_data1.append(pixels1[i + j])
            local_data2.append(pixels2[i + j])
            local_data3.append(pixels3[i + j])

        train_data1.append(local_data1)
        train_data2.append(local_data2)
        train_data3.append(local_data3)
        i = i + sliding
        del local_data1
        del local_data2
        del local_data3

    train_data1 = np.array(train_data1)
    train_data2 = np.array(train_data2)
    train_data3 = np.array(train_data3)
    print("train_data1.shape:", train_data1.shape)
    print("train_data2.shape:", train_data2.shape)
    print("train_data3.shape:", train_data3.shape)
    labels = kinect_75_labels[75]
    print("labels.unique:", np.unique(labels))
    print("labels.len:", labels.shape)
    
    frames_together = together
    sliding = slide
    
    print("frames_togher:", frames_together)

    slide_labels = []

    i = 0
    count = 0
    del_window = []
    while i + frames_together <= pixels1.shape[0]:
        local_data = []
        count0 = 0
        count1 = 0

        count2 = 0
        count3 = 0
        count4 = 0
        count5 = 0
        count6 = 0

        for j in range(frames_together):
            label = labels[i + j]
            if label == 0:
                count0 += 1
            if label == 1:
                count1 += 1
            if label == 2:
                count2 += 1

            if label == 3:
                count3 += 1
            if label == 4:
                count4 += 1
            if label == 5:
                count5 += 1
            if label == 6:
                count6 += 1

        judge = [count0, count1, count2, count3, count4, count5, count6]
        print("judge:", judge)
        #         kinds=["0","0_1","1_0","1_2","2_1","0_2","2_0"]
        kinds = [0, 1, 2, 3, 4, 5, 6]
        if np.max(judge) < frames_together *1:
            del_window.append(count)

        label = kinds[np.argmax(judge)]
        slide_labels.append(label)
        i = i + sliding
        count += 1
    slide_labels = np.array(slide_labels)

    np.unique(slide_labels)
    
    print(slide_labels)
#    print("slide_labels.shape:", np.unique(slide_labels))

    labels_uncoded = slide_labels
    sub_dirs = np.unique(slide_labels).tolist()
    #     sub_dirs=[0,1,2,3,4]
    train_label = one_hot_encoding(slide_labels, sub_dirs, categories=len(sub_dirs))
#     print("删除的window数量：", del_window)
    train_data1 = np.delete(train_data1, del_window, axis=0)
    train_data2 = np.delete(train_data2, del_window, axis=0)
    train_data3 = np.delete(train_data3, del_window, axis=0)
    train_label = np.delete(train_label, del_window, axis=0)
    labels_uncoded = np.squeeze(np.delete(labels_uncoded.reshape(-1, 1), del_window, axis=0))
#    print("labels_uncoded:", labels_uncoded)
    print("删除window数量后train_data1:", train_data1.shape)
    print("删除window数量后train_data2:", train_data2.shape)
    print("删除window数量后train_data3:", train_data3.shape)
    print("删除window数量后train_label:", train_label.shape)

    return data_pro1, train_data1, train_data2, train_data3, train_label, labels_uncoded
# place="./3_2dcnn_predict_data/"
# want=[10,10,10,12,2]
# point_data_pre,kinect_76_labels_final_pre=contain_all_files_data_labels(r"./predict/")
#
# data_pro_pre,data_pre1,data_pre2,data_pre3,label_pre,labels_uncoded_pre=data_to_picture(point_data_pre,kinect_76_labels_final_pre,want[0],want[1],want[2],want[3],want[4])
#
#
# path1=place+"predict"+"data_pre1"
# path2=place+"predict"+"data_pre2"
# path3=place+"predict"+"data_pre3"
# path4=place+"predict"
# if os.path.exists(path1)==False:
#     # print("./train/不存在，新建：")
#     os.makedirs(path1)
#     os.makedirs(path2)
#     os.makedirs(path3)
#     os.makedirs(path4)
# np.save(place+"predict"+"data_pre1",data_pre1)
# np.save(place+"predict"+"data_pre2",data_pre2)
# np.save(place+"predict"+"data_pre3",data_pre3)
# np.save(place+"predict",label_pre)


# In[2]:


for cut in np.arange(50, 60, 10).tolist():
    x_cut = cut
    z_cut = cut
    for y_cut in np.arange(30, 35, 5).tolist():
        y_cut=y_cut
        for t_f in np.arange(11, 13, 2).tolist():
            place="./predict_驗證不同疊加長度的文件夾/predict"+str(x_cut)+"_"+str(y_cut)+"_"+str(z_cut)+"_"+str(t_f)+"/"
            want=[x_cut,y_cut,z_cut,t_f,3]
            point_data_pre,kinect_76_labels_final_pre=contain_all_files_data_labels(r"./predict/")

            data_pro_pre,data_pre1,data_pre2,data_pre3,label_pre,labels_uncoded_pre=data_to_picture(point_data_pre,kinect_76_labels_final_pre,want[0],want[1],want[2],want[3],want[4])


            # path1=place+"predict"+"data_pre1/"
            # path2=place+"predict"+"data_pre2/"
            # path3=place+"predict"+"data_pre3/"
            # path4=place+"predict"
            path1=place
            name1=path1+"predict"+"data_pre1"
            name2 = path1 + "predict" + "data_pre2"
            name3 = path1 + "predict" + "data_pre3"
            name4 = path1 + "predict"
            # path2=place
            # path3=place
            # path4=place
            if os.path.exists(path1)==False:
                # print("./train/不存在，新建：")
                os.makedirs(path1)
                # os.makedirs(path2)
                # os.makedirs(path3)
                # os.makedirs(path4)
            np.save(name1,data_pre1)
            np.save(name2,data_pre2)
            np.save(name3,data_pre3)
            np.save(name4,label_pre)
            
print("搞定！")


# In[ ]:




