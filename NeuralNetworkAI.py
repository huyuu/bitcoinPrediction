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

from plotGraph import plot
from PreprocessingWorker import PreprocessingWorker

modelFilePath = "nnmodel.h5"
# path = "BTC-JPY.csv"


class NeuralNetworkAI():
    def __init__(self, zoneLength=20, determinateSigma=1.0, parameters=['BBPosition', 'RSI14']):
        self.zoneLength = zoneLength  # days
        self.determinateSigma = determinateSigma  # 1.0 * sigma
        self.parameters = parameters
        self.model = self.__buildModel()


    def trainSpanData(self, span='15MIN', testifyRadio=0.2, accuracyNeeded=0.4):
        dirName = './LabeledData'
        if os.path.exists(dirName):
            pass
        else:
            os.mkdir(dirName)

        if span == '15MIN':
            path = f'{dirName}/{span}.csv'
            data = pd.read_csv(path).dropna().reset_index(drop=True)
            self.__train(data, testifyRadio, accuracyNeeded)


    def __train(self, data, testifyRadio=0.2, accuracyNeeded=0.4):
        # fetch data
        # data = pd.read_csv(path).dropna().reset_index(drop=False)
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
            self.model.fit(trainSamples, trainLabels.ravel(), epochs=30)
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

        self.model = kr.models.load_model(modelFilePath)
        probabilities = self.model.predict(samples)
        predictedLabels = nu.argmax(probabilities, axis=1)
        data = pd.DataFrame(nu.concatenate([samples, predictedLabels.reshape(-1, 1), probabilities[:, 0].reshape(-1, 1), probabilities[:, 1].reshape(-1, 1), probabilities[:, 2].reshape(-1, 1)], axis=1), columns=['BBPosition', 'RSI14', 'ClassLabel', 'Label0_prob', 'Label1_prob', 'Label2_prob'])
        print(data)
        _0samples = data[data.ClassLabel == 0]
        _1samples = data[data.ClassLabel == 1]
        _2samples = data[data.ClassLabel == 2]

        if shouldShowImmediately:
            pl.imshow(probabilities, extent=(_BBsamples.min(), _BBsamples.max(), _RSIsamples.min(), _RSIsamples.max()), origin='lower', cmap='RdBu_r')
            # pl.imshow(probabilities, extent=(_BBsamples.min(), _BBsamples.max(), _RSIsamples.min(), _RSIsamples.max()), origin='lower', cmap='viridis')
            # pl.imshow(probabilities, origin='lower', cmap='viridis')
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


    def predictCurrentSituation(self, data, shouldTrainAgain=False, shouldShowGraph=True):
        # general constants
        dirName = './Prediction'
        if os.path.exists(dirName):
            pass
        else:
            os.mkdir(dirName)

        # fetch sample
        # data['RSI14'] = ta.RSI(data['Close'], timeperiod=14)
        # data['BB+1sigma'], data['BBmiddle'], data['BB-1sigma'] = ta.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        # data['Sigma'] = (data['BB+1sigma'] - data['BBmiddle'])/2
        # data['BBPosition'] = (data['Close'] - data['BBmiddle'])/data['Sigma']
        sample = data[self.parameters].values.reshape(1, -1)

        # if should train again, run the train. Otherwise, use previous model
        if shouldTrainAgain or not os.path.exists(modelFilePath):
            self.trainSpanData()
        else:
            self.model = kr.models.load_model(modelFilePath)

        # run predict
        backgroundProbs, _BBsamplesGrid, _RSIsamplesGrid = self.showModel(shouldShowImmediately=shouldShowGraph)
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
        else:
            fig = pl.figure()
            pl.imshow(backgroundProbs, extent=(_BBsamplesGrid.min(), _BBsamplesGrid.max(), _RSIsamplesGrid.min(), _RSIsamplesGrid.max()), origin='lower', cmap='RdBu_r')
            pl.scatter(sample.ravel()[0], sample.ravel()[1], c='Black')
            pl.xlabel(r'BB Position [$\Sigma$]')
            pl.ylabel('RSI 14')
            fig.savefig(f'{dirName}/2DImage.png')

            fig = pl.figure()
            ax = pl.axes(projection='3d')
            ax.plot_surface(_BBsamplesGrid, _RSIsamplesGrid, backgroundProbs[:, 0].reshape(_points, _points), cmap='Blues')
            ax.scatter3D(sample.ravel()[0], sample.ravel()[1], probabilities[0], c='Black', s=100)
            fig.savefig(f'{dirName}/lowClassProbability.png')

            fig = pl.figure()
            ax = pl.axes(projection='3d')
            ax.plot_surface(_BBsamplesGrid, _RSIsamplesGrid, backgroundProbs[:, 1].reshape(_points, _points), cmap='gray')
            ax.scatter3D(sample.ravel()[0], sample.ravel()[1], probabilities[1], c='Black', s=100)
            fig.savefig(f'{dirName}/middleClassProbability.png')

            fig = pl.figure()
            ax = pl.axes(projection='3d')
            ax.plot_surface(_BBsamplesGrid, _RSIsamplesGrid, backgroundProbs[:, 2].reshape(_points, _points), cmap='Reds')
            ax.scatter3D(sample.ravel()[0], sample.ravel()[1], probabilities[2], c='Black', s=100)
            fig.savefig(f'{dirName}/highClassProbability.png')

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
            kr.layers.Dense(10, activation='relu'),
            kr.layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model


if __name__ == '__main__':
    # worker = PreprocessingWorker()
    # worker.process()

    ai = NeuralNetworkAI()
    # ai.trainSpanData()
    # ai.showModel(shouldShowImmediately=True)
    data = pd.read_csv('./LabeledData/15MIN.csv').tail(1)
    ai.predictCurrentSituation(data, shouldShowGraph=True)
    # ai.predictToday(shouldTrainAgain=False, shouldShowGraph=True)
    # plot()
