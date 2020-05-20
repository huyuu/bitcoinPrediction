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
import talib as ta
from tensorflow import keras as kr
import seaborn as sns

from plotGraph import plot
from PreprocessingWorker import PreprocessingWorker

modelFilePath = "nnmodel.h5"
path = "BTC-JPY.csv"


class NeuralNetworkAI():
    def __init__(self, zoneLength=30, determinateSigma=1.0, parameters=['BBPosition', 'RSI14']):
        self.zoneLength = zoneLength  # days
        self.determinateSigma = determinateSigma  # 1.0 * sigma
        self.parameters = parameters
        self.model = self.__buildModel()


    def trainFromHistoryData(self, testifyRadio=0.2, accuracyNeeded=0.38):
        # fetch data
        data = pd.read_csv(path).dropna().reset_index(drop=False)
        # set samples
        testSamplesAmount = int(data.index.values.shape[0] * testifyRadio)
        trainSamplesAmount = data.index.values.shape[0] - testSamplesAmount
        trainSamples = data[self.parameters].values[:trainSamplesAmount, :]
        trainLabels = data['ClassLabel'].values[:trainSamplesAmount].reshape(-1, 1)
        testSamples = data[self.parameters].values[trainSamplesAmount:, :]
        testLabels = data['ClassLabel'].values[trainSamplesAmount:].reshape(-1, 1)

        # start training
        print('Start training model ...')
        _start = dt.datetime.now()

        accuracy = 0
        while accuracy < accuracyNeeded:
            self.model.fit(trainSamples, trainLabels.ravel(), epochs=15)
            loss, accuracy = self.model.evaluate(testSamples, testLabels.ravel())
            if accuracy < accuracyNeeded:
                print('Accuracy not enough, try again ...')
                self.model = self.__buildModel()

        self.model.save(modelFilePath)
        timeCost = (dt.datetime.now() - _start).total_seconds()
        print(f'Model training ends with accuracy {accuracy}. (time cost: {timeCost} seconds)')


    def showModel(self, shouldShowImmediately=True):
        points = 100
        _BBsamples = nu.linspace(-3, 3, points)
        _RSIsamples = nu.linspace(0, 1, points)
        _BBsamplesGrid, _RSIsamplesGrid = nu.meshgrid(_BBsamples, _RSIsamples, indexing='ij')
        samples = nu.concatenate([_BBsamplesGrid.reshape(-1, 1), _RSIsamplesGrid.reshape(-1, 1)], axis=1)

        probabilities = self.model.predict(samples)
        predictedLabels = nu.argmax(probabilities, axis=1)
        data = pd.DataFrame(nu.concatenate([samples, predictedLabels.reshape(-1, 1), probabilities[:, 0].reshape(-1, 1), probabilities[:, 1].reshape(-1, 1), probabilities[:, 2].reshape(-1, 1)], axis=1), columns=['BBPosition', 'RSI14', 'ClassLabel', 'Label0_prob', 'Label1_prob', 'Label2_prob'])
        _0samples = data[data.ClassLabel == 0]
        _1samples = data[data.ClassLabel == 1]
        _2samples = data[data.ClassLabel == 2]

        if shouldShowImmediately:
            # pl.imshow(probabilities, extent=(_BBsamples.min(), _BBsamples.max(), _RSIsamples.min(), _RSIsamples.max()), origin='lower', cmap='RdBu_r')
            # pl.imshow(probabilities, extent=(_BBsamples.min(), _BBsamples.max(), _RSIsamples.min(), _RSIsamples.max()), origin='lower', cmap='viridis')
            pl.imshow(probabilities, origin='lower', cmap='viridis')
            pl.show()
        return probabilities, _BBsamplesGrid, _RSIsamplesGrid

        # ax = pl.axes(projection='3d')
        # ax.plot_surface(_BBsamplesGrid, _RSIsamplesGrid, data['Label0_prob'].values.reshape(points, points), cmap='Blues')
        # pl.show()
        #
        # fig = pl.figure()
        # ax = pl.axes(projection='3d')
        # ax.plot_surface(_BBsamplesGrid, _RSIsamplesGrid, data['Label1_prob'].values.reshape(points, points), cmap='gray')
        # pl.show()
        #
        # fig = pl.figure()
        # ax = pl.axes(projection='3d')
        # ax.plot_surface(_BBsamplesGrid, _RSIsamplesGrid, data['Label2_prob'].values.reshape(points, points), cmap='Reds')
        # pl.show()


    def predictToday(self, shouldTrainAgain=False, shouldShowGraph=True):
        # check if data is up-to-date
        data = pd.read_csv(path).tail(1)
        now = dt.datetime.utcnow()
        yesterdayString = dt.datetime(now.year, now.month, now.day - 1).strftime("%Y-%m-%d")
        if data['Date'].values != yesterdayString or all([ parameter in data.dropna(axis=1).columns.values for parameter in self.parameters ]):
            worker = PreprocessingWorker()
            worker.process
            sample = pd.read_csv(path).tail(1)[self.parameters].values.reshape(1, -1)
            del worker, data
        else:
            sample = data.tail(1)[self.parameters].values.reshape(1, -1)
            del data

        # if should train again, run the train. Otherwise, use previous model
        if shouldTrainAgain or not os.path.exists(modelFilePath):
            self.trainFromHistoryData()
        else:
            self.model = kr.models.load_model(modelFilePath)

        # run predict
        backgroundProbs, _BBsamplesGrid, _RSIsamplesGrid = self.showModel(shouldShowImmediately=True)
        _points = _BBsamplesGrid.shape[0]

        probabilities = self.model.predict(sample)[0, :]

        if shouldShowGraph:
            pl.imshow(backgroundProbs, extent=(_BBsamplesGrid.min(), _BBsamplesGrid.max(), _RSIsamplesGrid.min(), _RSIsamplesGrid.max()), origin='lower', cmap='RdBu_r')
            pl.scatter(sample.ravel()[0], sample.ravel()[1], c='Black')
            pl.xlabel(r'BB Position [$\Sigma$]')
            pl.ylabel('RSI 14')
            pl.show()

            ax = pl.axes(projection='3d')
            ax.plot_surface(_BBsamplesGrid, _RSIsamplesGrid, backgroundProbs[:, 0].reshape(_points, _points), cmap='Blues')
            ax.scatter3D(sample.ravel()[0], sample.ravel()[1], probabilities[0], c='Black', s=100)
            pl.show()

            ax = pl.axes(projection='3d')
            ax.plot_surface(_BBsamplesGrid, _RSIsamplesGrid, backgroundProbs[:, 1].reshape(_points, _points), cmap='gray')
            ax.scatter3D(sample.ravel()[0], sample.ravel()[1], probabilities[1], c='Black', s=100)
            pl.show()

            ax = pl.axes(projection='3d')
            ax.plot_surface(_BBsamplesGrid, _RSIsamplesGrid, backgroundProbs[:, 2].reshape(_points, _points), cmap='Reds')
            ax.scatter3D(sample.ravel()[0], sample.ravel()[1], probabilities[2], c='Black', s=100)
            pl.show()

        label0Position = (probabilities[0] - backgroundProbs[:, 0].ravel().mean())/backgroundProbs[:, 0].ravel().std()
        label1Position = (probabilities[1] - backgroundProbs[:, 1].ravel().mean())/backgroundProbs[:, 1].ravel().std()
        label2Position = (probabilities[2] - backgroundProbs[:, 2].ravel().mean())/backgroundProbs[:, 2].ravel().std()

        # reference: https://note.nkmk.me/python-format-zero-hex/
        print('Should buy?         -> probability: {:.1f}%, {:+.1f}% Sigma from mean.'.format(probabilities[0]*100.0, label0Position*100.0))
        print('Dont have an idea?  -> probability: {:.1f}%, {:+.1f}% Sigma from mean.'.format(probabilities[1]*100.0, label1Position*100.0))
        print('Should sell?        -> probability: {:.1f}%, {:+.1f}% Sigma from mean.'.format(probabilities[2]*100.0, label2Position*100.0))


    def __buildModel(self):
        model = kr.models.Sequential([
            kr.layers.Flatten(input_shape=(len(self.parameters),)),
            kr.layers.Dense(10, activation='relu'),
            kr.layers.Dense(10, activation='relu'),
            kr.layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model


if __name__ == '__main__':
    # worker = PreprocessingWorker()
    # worker.process()

    ai = NeuralNetworkAI()
    ai.predictToday(shouldTrainAgain=False, shouldShowGraph=True)
    # plot()
