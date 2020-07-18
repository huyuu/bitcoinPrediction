from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as nu
import pandas as pd
import multiprocessing as mp
import sklearn as sk
from sklearn import gaussian_process as gp
import pickle
from matplotlib import pyplot as pl
import datetime as dt
import os
from tensorflow import keras as kr
import seaborn as sns
from PreprocessingWorker import PreprocessingWorker


class CNNAI():
    def __init__(self):
        self.resolution = int(24*4)
        self.timeSpreadPast = int(24*2)
        self.model = self.__buildModel()
        self.modelPath = "cnnmodel.h5"


    def train(self, testifyRadio=0.2, accuracyNeeded=0.4):
        _start = dt.datetime.now()
        # check if dir exists
        dirName = './LabeledData'
        if os.path.exists(dirName):
            pass
        else:
            os.mkdir(dirName)
        # get labeled span data
        span = '15MIN'
        path = f'{dirName}/{span}.csv'
        data = pd.read_csv(path).dropna().reset_index(drop=True)
        print('class probability:\n{}%'.format(data.groupby('LabelCNNPost1').size() / float(data.index.values.ravel().shape[0]) * 100))
        graphAmount = len(data['Date'].values.ravel()[self.timeSpreadPast:-300])
        graphData = nu.zeros((graphAmount, self.timeSpreadPast, self.resolution), dtype=nu.int)
        # get graph
        for i, dateString in enumerate(data['Date'].values.ravel()[self.timeSpreadPast:-300]):
            graphName = dateString.split('.')[0].replace('T', '_').replace(':', '-')
            _graphData = pd.read_csv(f'{dirName}/graphData/{graphName}.csv', index_col=0)
            graphData[i, :, :] = _graphData.values.reshape(self.timeSpreadPast, self.resolution)
            # print(f'{i} of {graphAmount}')
        graphData = nu.random.permutation(graphData)
        graphData = graphData.reshape(-1, self.timeSpreadPast, self.resolution, 1)
        # start training precedure. First, preprocessing
        testSamplesAmount = int(graphData.shape[0] * testifyRadio)
        trainSamplesAmount = int(graphData.shape[0] - testSamplesAmount)
        trainSamples = graphData[:trainSamplesAmount, :, :, :]
        trainLabels = data['LabelCNNPost1'].values[self.timeSpreadPast:int(self.timeSpreadPast+trainSamplesAmount)].reshape(-1, 1)
        testSamples = graphData[trainSamplesAmount:, :, :, :]
        testLabels = data['LabelCNNPost1'].values[int(self.timeSpreadPast+trainSamplesAmount):-300].reshape(-1, 1)
        print('Start training model ...')
        # Second, train until accuracy needed is achieved
        accuracy = 0
        while accuracy < accuracyNeeded:
            self.model.fit(trainSamples, trainLabels.ravel(), epochs=10)
            loss, accuracy = self.model.evaluate(testSamples, testLabels.ravel())
            if accuracy < accuracyNeeded:
                print('Accuracy not enough, try again ...')
                self.model = self.__buildModel()
        # Third, save model.
        self.model.save(self.modelPath)
        timeCost = (dt.datetime.now() - _start).total_seconds()
        print(f'Model training ends with accuracy {accuracy}. (time cost: {timeCost} seconds)')


    def __buildModel(self):
        model = kr.models.Sequential([
            kr.layers.Conv2D(filters=64, kernel_size=5, activation='relu', input_shape=(self.timeSpreadPast, self.resolution, 1)),
            kr.layers.MaxPooling2D(pool_size=(3, 3)),
            kr.layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=(self.timeSpreadPast, self.resolution, 1)),
            kr.layers.MaxPooling2D(pool_size=(2, 2)),
            kr.layers.Dropout(0.25),
            kr.layers.Flatten(),
            kr.layers.Dense(50, activation='relu'),
            kr.layers.Dropout(0.25),
            kr.layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model



if __name__ == '__main__':
    worker = PreprocessingWorker()
    cnnModel = CNNAI()

    worker.processShortermHistoryData(span='15MIN', resolution=cnnModel.resolution, timeSpreadPast=cnnModel.timeSpreadPast)
    cnnModel.train()
