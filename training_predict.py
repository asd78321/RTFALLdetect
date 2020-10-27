#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !/usr/bin/env python
# coding: utf-8

# In[1]:


# !/usr/bin/env python
# coding: utf-8

# In[1]:
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
import scipy.io as scio
import os
import glob
import time
import csv

rand_seed = 1
from numpy.random import seed

seed(rand_seed)

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, Activation
from keras.layers.core import Permute, Reshape
from keras import backend as K

from keras import optimizers
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, Bidirectional, TimeDistributed
from sklearn.model_selection import train_test_split
from keras.models import load_model

from sklearn.metrics import confusion_matrix
# In[2]:
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, Activation
from keras.layers.core import Permute, Reshape
from keras import backend as K

from keras import optimizers
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, Bidirectional, TimeDistributed
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from keras.layers import Conv2D, Dense, MaxPooling2D, concatenate, Flatten, TimeDistributed, Dropout
from keras import Input, Model
from keras.utils import plot_model

from sklearn.preprocessing import label_binarize

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = plot_confusion_matrix

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #     classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)
from random import shuffle


def file_image_generator(inputPath1, inputPath2, bs, file_list1):
    #     shuffle(file_list1)
    index = 0
    while True:
        train_data1 = []
        train_data2 = []
        train_labels = []
        while len(train_data1) < bs:
            if index >= len(file_list1):
                index = 0
            temp1 = np.load(inputPath1 + file_list1[index]).reshape(want[3], want[1], want[0], 1)
            temp2 = np.load(inputPath2 + file_list1[index]).reshape(want[3], want[1], want[2], 1)
            train_data1.append(temp1)
            train_data2.append(temp2)
            labels_temp = int(file_list1[index].split(".")[1][-1])
            train_labels.append(labels_temp)

            index += 1

        #         sub_dirs=[0,1,2,3,4,5,6]
        #         train_label = one_hot_encoding(np.array(train_labels), sub_dirs, categories=7)
        train_label = label_binarize(np.array(train_labels), classes=[0, 1, 2, 3, 4, 5, 6])

        yield ([np.array(train_data1), np.array(train_data2)], train_label)




from keras.layers import Conv2D, Dense, MaxPooling2D, concatenate, Flatten, TimeDistributed, Dropout
from keras import Input, Model
from keras.utils import plot_model


def make_model(want):
    input1_ = Input(shape=(want[3], want[1], want[0], 1), name='input1')

    input2_ = Input(shape=(want[3], want[1], want[2], 1), name='input2')

    x1 = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), name="conv1a", padding="same", activation="relu"))(input1_)
    # 2nd layer group
    x1 = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), name="conv1b", padding="same", activation="relu"))(x1)

    x1 = TimeDistributed(MaxPooling2D(name="pool1", strides=(2, 2), pool_size=(2, 2), padding="valid"))(x1)

    # 3rd layer group
    x1 = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), name="conv2a", padding="same", activation="relu"))(x1)
    x1 = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), name="conv2b", padding="same", activation="relu"))(x1)
    x1 = TimeDistributed(MaxPooling2D(strides=(2, 2), pool_size=(2, 2), name="pool2", padding="valid"))(x1)

    x1 = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), name="conv2a", padding="same", activation="relu"))(x1)
    x1 = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), name="conv2b", padding="same", activation="relu"))(x1)
    x1 = TimeDistributed(MaxPooling2D(strides=(2, 2), pool_size=(2, 2), name="pool2", padding="valid"))(x1)

    # 1st layer group
    x2 = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), name="conv1a", padding="same", activation="relu"))(input2_)
    # 2nd layer group
    x2 = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), name="conv1b", padding="same", activation="relu"))(x2)

    x2 = TimeDistributed(MaxPooling2D(name="pool1", strides=(2, 2), pool_size=(2, 2), padding="valid"))(x2)

    # 3rd layer group
    x2 = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), name="conv2a", padding="same", activation="relu"))(x2)
    x2 = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), name="conv2b", padding="same", activation="relu"))(x2)
    x2 = TimeDistributed(MaxPooling2D(strides=(2, 2), pool_size=(2, 2), name="pool2", padding="valid"))(x2)

    x2 = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), name="conv2a", padding="same", activation="relu"))(x2)
    x2 = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), name="conv2b", padding="same", activation="relu"))(x2)
    x2 = TimeDistributed(MaxPooling2D(strides=(2, 2), pool_size=(2, 2), name="pool2", padding="valid"))(x2)

    x1 = TimeDistributed(Flatten(), name="flatten1")(x1)
    x2 = TimeDistributed(Flatten(), name="flatten2")(x2)

    x = concatenate([x1, x2], axis=2)

    x = Dropout(0.5)(x)

    x = Flatten()(x)

    x = Dropout(0.5)(x)

    x = Dense(256, activation='relu')(x)

    x = Dense(64, activation='relu')(x)

    x = Dense(7)(x)
    output_ = Activation('softmax')(x)
    #     output_=Dense(7, activation='softmax', name = 'output')(x)
    model = Model(inputs=[input1_, input2_], outputs=[output_])
    #     model.summary()
    return model


def decode(datum):
    return (np.argmax(datum, axis=1))



draw_model_val_acc = dict()
model_val_acc = dict()
for cut in np.arange(50, 60, 10).tolist():
    x_cut = cut
    z_cut = cut
    for y_cut in np.arange(30, 35, 5).tolist():
        y_cut = y_cut
        for t_f in np.arange(11, 13, 2).tolist():
            want = [x_cut, y_cut, z_cut, t_f, 2]
            train_data_rotate_test ="./test_2d_change/test" + str(x_cut) + "_" + str(y_cut) + "_" + str(z_cut) + "_"+str(t_f)+"/"
            train_data_rotate_12 ="./train_2d_change/train" + str(x_cut) + "_" + str(y_cut) + "_" + str(z_cut) + "_"+str(t_f)+"/"
            import random
            import os

            rate = 0.2
            filedir = train_data_rotate_12 + "/xoy/"
            path1 = train_data_rotate_12 + 'xoy/'
            path2 = train_data_rotate_12 + 'yoz/'
            all_list = os.listdir(filedir)
            shuffle(all_list)
            filenum = len(all_list)

            picknum = int(filenum * rate)
            validation_list = random.sample(all_list, picknum)
            train_list = list(set(all_list).difference(set(validation_list)))
            shuffle(train_list)

            filedir_list = train_data_rotate_test + "/xoy/"
            test_list = os.listdir(filedir_list)
            shuffle(test_list)
            path3 = train_data_rotate_test + 'xoy/'
            path4 = train_data_rotate_test + 'yoz/'

            label_pre_recoded = []
            for i in test_list:
                # print(i)
                labels_this = int(i.split(".")[1][-1])
                label_pre_recoded.append(labels_this)
            NUM_TRAIN_IMAGES = len(train_list)
            NUM_VALIDATION_IMAGES = len(validation_list)
            NUM_TEST_IMAGES = len(test_list)
            print("訓練集長度：", NUM_TRAIN_IMAGES)
            print("驗證集長度：", NUM_VALIDATION_IMAGES)
            print("測試集長度：", NUM_TEST_IMAGES)

            NUM_EPOCHS = 15
        
            BS = 23

            trainGen = file_image_generator(path1, path2, BS, train_list)
            validationGen = file_image_generator(path1, path2, BS, validation_list)
            #             testGen = file_image_generator_test(path3,path4,BS,test_list)
            print("len(train):", NUM_TRAIN_IMAGES)
            print("len(validation):", NUM_VALIDATION_IMAGES)
            print("len(test):", NUM_TEST_IMAGES)

            checkpoint_model_path1 = "./dual_view_cnn_change_11.h5"

            checkpoint1 = ModelCheckpoint(checkpoint_model_path1, monitor='val_loss', verbose=1, save_best_only=True,
                                          mode='min')

            callbacks_list = [checkpoint1]
            adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)
            # opt = SGD(lr=1e-2, momentum=0.9, decay=1e-2 / NUM_EPOCHS)
            #         want=[10,10,10,12,2]
            model = make_model(want)
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=adam,
                          metrics=['accuracy'])

            H1 = model.fit_generator(
                trainGen,
                steps_per_epoch=NUM_TRAIN_IMAGES // BS,
                validation_data=validationGen,
                validation_steps=NUM_VALIDATION_IMAGES // BS,
                epochs=NUM_EPOCHS,
                verbose=1,
                callbacks=callbacks_list)
            print("此時的切割：", want)

            print("best accuracy:", max(H1.history["val_accuracy"]))
            BS = 31
            end_index = NUM_TEST_IMAGES // BS * BS
            print("end_index:", end_index)
            testGen = file_image_generator(path3, path4, BS, test_list)
            y_pre_another = model.predict_generator(testGen, steps=NUM_TEST_IMAGES // BS, verbose=1)
            print(y_pre_another)
            y_pre_another = decode(y_pre_another)
            print(y_pre_another)
            print(y_pre_another.shape)

            target_names = ["move", "st_sit", "sit_st", "sit_lie", "lie_sit", "fall", "get_up"]
            from sklearn.metrics import accuracy_score

            print("精確度：\n", accuracy_score(np.array(label_pre_recoded[:end_index]), y_pre_another))

            precision_this = precision_score(np.array(label_pre_recoded[:end_index]), y_pre_another, average='macro')

            print(precision_this)
            recall_score_this = recall_score(np.array(label_pre_recoded[:end_index]), y_pre_another, average='macro')

            print(recall_score_this)
            f1_score_this = f1_score(np.array(label_pre_recoded[:end_index]), y_pre_another, average='macro')
            print(f1_score_this)

            from sklearn.metrics import accuracy_score

            class_names = ["move", "st_sit", "sit_st", "sit_lie", "lie_sit", "fall", "get_up"]
            print("cnn_的精確度：\n", accuracy_score(label_pre_recoded[:end_index], y_pre_another))
            plot_confusion_matrix(label_pre_recoded[:end_index], y_pre_another, classes=class_names, normalize=True,
                                  title='cnn_3')
            plt.savefig("./dual_view_cnn_change_11n.pdf")
            np.save("./dual_view_cnn_change_11n_history", H1.history)





plt.plot(H1.history["loss"])
plt.plot(H1.history["accuracy"])







