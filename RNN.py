"""
Title: Time series forecasting with regression for the Air Prediction in Madrid
Author: Myrthe Moring
Date: 18/12/2018

Description:
Time series forecasting with regression is an important tool for predicting time
serie sequences, in this case air quality.
The pollution is measured as the number of particulates of 20 micrometers
in air (PM20 ug/m3).

Since it is a regression problem, the measurements are MSE and R2.
"""

from __future__ import print_function
import numpy as np
import math
import pandas
import matplotlib.pyplot as plt
import os
import json
import argparse
from time import time
import time
import datetime

import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, GRU, SimpleRNN
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def lagged(data, lag=1, ahead=0):
    """
    Returns a vector with columns that are the steps of the lagged time series
    Last column is the value to predict

    Because arrays start at 0, Ahead starts at 0 but actually means one step ahead
    """
    lvect = []
    for i in range(lag):
        lvect.append(data[i: -lag - ahead + i])
    lvect.append(data[lag + ahead:])

    return np.stack(lvect, axis=1)

def architecture(neurons, drop, nlayers, activation, activation_r, rnntype, impl=1):
    """
    RNN architecture:
    1. n RNN layers with as input size the length of the training data and with
    1 attribute per element in the sequence,
    3. Dense layer to compute the output for the regression
    """
    RNN = LSTM
    model = Sequential()

    if nlayers == 1:
        model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), implementation=impl,
                      recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r))
    else:
        model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), implementation=impl,
                      recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                      return_sequences=True))
        for i in range(1, nlayers - 1):
            model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl,
                          activation=activation, recurrent_activation=activation_r, return_sequences=True))
        model.add(RNN(neurons, recurrent_dropout=drop, activation=activation,
                      recurrent_activation=activation_r, implementation=impl))

    model.add(Dense(1))
    return model

def read_data(begin, end):
    """
    Read the data from the csv file given the begin and end date.
    Return: pandas dataframe
    """
    if end == 0:
        return pandas.read_csv("madrid_" + str(begin) + ".csv", engine='python')
    data_years = []
    for year in range(begin,end):
        data_year = pandas.read_csv("madrid_" + str(year) + ".csv")
        data_years.append(data_year)
    df = pandas.concat(data_years, axis=0)
    return df.reset_index(drop=True)

def split_data(data_metrics, split_size, ahead, lag):
    """
    Split the data based on the split size (80/20)
    """
    datasize = int(len(data_metrics) * split_size)
    testsize = len(data_metrics) - datasize

    train = lagged(data_metrics[:datasize, :], lag=lag, ahead=0)
    test = lagged(data_metrics[datasize:datasize + testsize, 0].reshape(-1, 1), lag=lag, ahead=0)

    return train[:, :lag], train[:, -1:, 0], test[:, :lag], test[:, -1:, 0]

def type_data(type, scaler):
    """
    Preprocessing the data given the type:
    - removing Nan's
    - normalizing
    """
    data = (madrid_data[type])
    data = pandas.DataFrame(data)
    data = data.interpolate(method='nearest')
    data = data.fillna(data.mean())
    data = data.astype('float32')
    data_metrics = scaler.fit_transform(data)
    return data_metrics

if __name__ == '__main__':

    verbose, impl = 0, 1

    # Random seed for reproducibility
    np.random.seed(7)

    begin_year, end_year = 2016, 2018
    duration = end_year-begin_year
    madrid_data = read_data(begin_year,end_year)

    columns = ['BEN', 'CO', 'NO_2', 'SO_2']

    type = 'BEN'
    scaler = StandardScaler()
    data_metrics = type_data(type, scaler)

    # plt.plot(data_metrics)
    # plt.xticks([])
    # plt.show()

    split_size, ahead, lag = 0.80, 1, 4
    train_x, train_y, test_x, test_y = split_data(data_metrics, split_size, ahead, lag)

    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    # RNN type = LSTM, GRU or RNNsimple
    RNN_type = "LSTM"

    """ Model architecture: """
    model = architecture(neurons=64,
                         drop=0.1,
                         nlayers=2,
                         activation="tanh",
                         activation_r="hard_sigmoid",
                         rnntype=RNN_type,
                         impl=impl)

    """ Training """
    batch_size = 512
    nr_epochs = 16

    optimizer = RMSprop(lr=0.001)

    start = time.time()

    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.fit(train_x, train_y,
                batch_size=batch_size,
                epochs=nr_epochs,
                validation_data=(test_x, test_y),
                verbose=verbose)

    end = time.time()
    print("Duration", end - start)

    """ Results: """
    mse_score = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=0)
    score_model = mean_squared_error(test_y[ahead:], test_y[0:-ahead])
    print('MSE test= ', mse_score)
    print('MSE test persistence =', score_model)
    print('Deviation with baseline: ', (score_model - mse_score)/score_model)

    test_yp = model.predict(test_x, batch_size=batch_size, verbose=0)
    train_yp = model.predict(train_x, batch_size=batch_size, verbose=0)
    train_p = scaler.inverse_transform(train_yp)
    trainY = scaler.inverse_transform([train_y])
    test_p = scaler.inverse_transform(test_yp)
    testY = scaler.inverse_transform([test_y])

    r2test = r2_score(test_y, test_yp)
    r2_model = r2_score(test_y[ahead:, 0], test_y[0:-ahead, 0])
    print('R2 test= ', r2test)
    print('R2 test persistence =', r2_model)
    print('Deviation with baseline: ', (r2test-r2_model)/r2test)

    " Test + training data plot "
    train_plot = np.empty_like(data_metrics)
    train_plot[:, :] = np.nan
    train_plot[ahead:len(train_p)+ahead, :] = train_p
    test_plot = np.empty_like(data_metrics)
    test_plot[:, :] = np.nan
    test_plot[len(train_p)+(ahead*2)+3:len(data_metrics)-3, :] = test_p
    plt.plot(scaler.inverse_transform(data_metrics))
    plt.plot(train_plot)
    plt.plot(test_plot)
    plt.title(type + " " +  RNN_type)
    plt.show()
