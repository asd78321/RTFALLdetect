#!/usr/bin/python
# coding:utf-8

import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
from keras import optimizers
import pickle


def load_data(data_path):
    file_name_list = os.listdir(data_path)
    train_data = np.array([])
    train_label = np.array([])
    for file_name_index in range(len(file_name_list)):
        data = np.load(data_path + "/{}".format(file_name_list[file_name_index]))
        label = np.zeros([len(data), 7])
        label[:, file_name_index] = 1
        if train_data.size != 0:
            train_data = np.concatenate((train_data, data))
            train_label = np.concatenate((train_label, label))
        else:
            train_data = data
            train_label = label
        # print(np.shape(train_data), np.shape(train_label))
        # print(label[0])
    # train_data = train_data.reshape([len(train_data), 70])
    print(np.shape(train_data), np.shape(train_label))

    return train_data, train_label


def model_LSTM(X_train, y_train):
    learning_rate = 0.0001
    batch_size = [8]
    # Initialising the RNN
    model = Sequential()
    #
    # Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units=140, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.1))
    #
    #     # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units=70, return_sequences=True))
    model.add(Dropout(0.1))

    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units=70, return_sequences=True))
    model.add(Dropout(0.1))
    #
    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units=70))
    model.add(Dropout(0.1))

    # Adding the output layer
    model.add(Dense(units=1, activation='relu'))

    # Compiling
    model.compile(optimizer='adam', loss='mse')

    # 進行訓練
    model.fit(X_train, y_train, epochs=30, batch_size=8)


def data_shuffle_and_split(train_data, train_label, training_size):
    shuffling_index = random.sample(range(train_data[:, 0].size), train_data[:, 0].size);
    x_train = train_data[shuffling_index, :]
    y_train = train_label[shuffling_index, :]

    x_test = x_train[int(training_size * y_train[:, 0].size):, :]
    y_test = y_train[int(training_size * y_train[:, 0].size):, :]

    x_train = x_train[0:int(training_size * y_train[:, 0].size)]
    y_train = y_train[0:int(training_size * y_train[:, 0].size)]

    print("Training_data_length:{}, Testing_data_length:{}".format(len(x_train), len(x_test)))
    return x_train, y_train, x_test, y_test


def model_dnn(X_train, Y_train):
    training_size = 0.9
    X_train, Y_train, X_test, Y_test = data_shuffle_and_split(X_train, Y_train, training_size)

    learn_rate = 0.0001
    batch_size = [10]
    model = Sequential()
    model.add(Dense(4096, activation='relu', input_dim=70))
    model.add(Dropout(0.2))
    model.add(Dense(2048, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(2048, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(2048, activation='relu'))
    # model.add(Dropout(0.15))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    optimizers.Adam(lr=0.0003)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.summary()
    best_weights_filepath = './model_cnn.h5'
    saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_accuracy', verbose=2,
                                    save_best_only=True, mode='auto')
    earlyStopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2, mode='auto')
    history = model.fit(X_train, Y_train, batch_size=8, validation_data=[X_test, Y_test], epochs=1000, verbose=1,
                        callbacks=[earlyStopping, saveBestModel])

    with open('trainHistoryDict.txt', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


def main():
    data_path = "./training_data_doppler"
    train_data, train_label = load_data(data_path)
    print("Doppler length:{}".format(len(train_data)))
    data_path = "./training_data_Height"
    train_data,train_label = load_data(data_path)
    print("Pointcloud length:{}".format(len(train_data)))
    # model_LSTM(train_data,train_label)
    # model_dnn(train_data, train_label)


if __name__ == "__main__":
    main()
