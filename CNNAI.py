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
from PreprocessingWorker import PreprocessingWorker, dateToString, stringToDate


class CNNAI():
    def __init__(self, model=None):
        self.resolution = int(24*8)
        self.timeSpreadPast = int(24*8)
        self.modelPath = "cnnmodel.h5"
        if model is None:
            self.model = self.__buildModel()
            self.isModelLoaded = False
        else:
            self.model = model
            self.isModelLoaded = True



    def train(self, testifyRadio=0.2, accuracyNeeded=0.45, trainEpochsEachTime=20):
        _start = dt.datetime.now()
        # check if dir exists
        dirName = './LabeledData'
        if os.path.exists(dirName):
            pass
        else:
            os.mkdir(dirName)
        # get labeled span data
        span = '15MIN'
        path = f'{dirName}/{span}/labeledData.csv'
        data = pd.read_csv(path).dropna().reset_index(drop=True)
        # latest data is unstable, should be dropped.
        data = data.drop(data.index[-self.timeSpreadPast:])
        data = data.reindex(nu.random.permutation(data.index))
        data = data.reset_index(drop=True)
        print('class probability:\n{}%'.format(data.groupby('LabelCNNPost1').size() / float(data.index.values.ravel().shape[0]) * 100))
        graphAmount = data.index.values.ravel().shape[0]
        graphData = nu.zeros((graphAmount, self.timeSpreadPast, self.resolution), dtype=nu.int)
        # get graph
        # for i, dateString in enumerate(data['Date'].values.ravel()):
        for i in data.index:
            dateString = data.loc[i, 'Date']
            # print(f'data[{i}] = {data.loc[i]}')
            graphName = dateString.split('.')[0].replace('T', '_').replace(':', '-')
            _graphData = pd.read_csv(f'{dirName}/{span}/graphData/{graphName}.csv', index_col=0)
            graphData[i, :, :] = _graphData.values.reshape(self.timeSpreadPast, self.resolution)
            # print(f'{i} of {graphAmount}')
        graphData = graphData.reshape(graphAmount, self.timeSpreadPast, self.resolution, 1)
        # start training precedure. First, preprocessing
        testSamplesAmount = int(graphData.shape[0] * testifyRadio)
        trainSamplesAmount = int(graphData.shape[0] - testSamplesAmount)
        trainSamples = graphData[:trainSamplesAmount, :, :, :]
        trainLabels = data['LabelCNNPost1'].values[:int(trainSamplesAmount)].reshape(-1, 1)
        testSamples = graphData[trainSamplesAmount:, :, :, :]
        testLabels = data['LabelCNNPost1'].values[trainSamplesAmount:].reshape(-1, 1)
        print('Start training model ...')
        # Second, train until accuracy needed is achieved
        accuracy = 0
        while accuracy < accuracyNeeded:
            self.model.fit(trainSamples, trainLabels.ravel(), epochs=trainEpochsEachTime)
            loss, accuracy = self.model.evaluate(testSamples, testLabels.ravel())
            if accuracy < accuracyNeeded:
                print('Accuracy not enough, try again ...')
                self.model = self.__buildModel()
        # Third, save model.
        self.model.save(self.modelPath)
        timeCost = (dt.datetime.now() - _start).total_seconds()
        print(f'Model training ends with accuracy {accuracy}. (time cost: {timeCost} seconds)')


    def showModelBreviation(self, graphDataDir, span='15MIN'):
        # get valid model
        self.__checkAndHandleLoadingModel()
        data = pd.read_csv('./LabeledData/{span}/labeledData.csv')
        # set target time
        # targetTime = dt.datetime(2020, 7, 23, 0, 0, 0)
        targetTime = stringToDate(data.loc[data.index.values.shape[0]-1, 'Date'])
        # get graph data
        graphDataName = targetTime.strftime('%Y-%m-%d_%H-%M-%S') + '.csv'
        graphDataPath = f'{graphDataDir}/{graphDataName}'
        # check if path exists.
        if not os.path.exists(graphDataPath):
            print(f"{graphDataPath} doesn't exist.")
        # get current data
        graphData = pd.read_csv(graphDataPath, index_col=0)
        plotData = nu.rot90(graphData.values)
        # if label available, get future data
        dateString = targetTime.strftime('%Y-%m-%d') + 'T' + targetTime.strftime('%H:%M:%S') + '.0000000Z'
        t = data.loc[data['Date'] == dateString, :].index.values[0]
        timeSpreadFuture = int(self.timeSpreadPast / 4)
        if int(t+timeSpreadFuture) in data.index.values:
            graphDataFutureName = data.loc[t+timeSpreadFuture, 'Date'].values[0].split('.')[0].replace('T', '_').replace(':', '-') + '.csv'
            graphDataFuture = pd.read_csv(f'{graphDataDir}/{graphDataFutureName}', index_col=0)
            plotData = nu.rot90(nu.concatenate([graphData.values, graphDataFuture.values[-timeSpreadFuture:, :]]))
        # get prediction
        prediction = self.model.predict(graphData.values.reshape(1, self.timeSpreadPast, self.resolution, 1))[0]

        if data.loc[data['Date'] == dateString, 'LabelCNNPost1'].values.ravel()[0] in nu.array([0, 1.0, 2.0]):
            terms = ['+', '0', '-']
            label = int(data.loc[data['Date'] == dateString, 'LabelCNNPost1'].values[0])
            pl.title('Prediction: +: {:.3g}%, 0: {:.3g}%, -: {:.3g}% (Actually {})'.format(prediction[0]*100, prediction[1]*100, prediction[2]*100, terms[label]), fontsize=26)
        else:
            pl.title('Prediction: +: {:.3g}%, 0: {:.3g}%, -: {:.3g}%'.format(prediction[0]*100, prediction[1]*100, prediction[2]*100), fontsize=26)
        pl.xlabel('Date', fontsize=22)
        pl.ylabel('Value', fontsize=22)
        pl.imshow(plotData, cmap = 'gray')
        pl.plot((graphData.shape[0]-1)*nu.ones(10), nu.linspace(0, graphData.shape[1]-1, 10), '--', c='Red')
        pl.show()


    def predictFromCurrentData(self, data, now, shouldSaveGraph, graphDataDir=None):
        self.__checkAndHandleLoadingModel()

        _minute = (now.minute // 15) * 15
        _t = dt.datetime(now.year, now.month, now.day, now.hour, _minute, 0)
        t = data.loc[data['DateTypeDate'] == _t, :].index.values
        # if data at now is not provided, break.
        if t.shape[0] == 0:
            print(f'Current data ({_t}) not provided, skip prediction.')
            return
        t = int(t[0])
        # if some middle data are missing, break.
        if data.loc[t, 'DateTypeDate'] - data.loc[t-self.timeSpreadPast, 'DateTypeDate'] >= dt.timedelta(minutes=(int(self.timeSpreadPast*1.5)*15)):
            # print(data.loc[t-self.timeSpreadPast, 'DateTypeDate'])
            # print(data.loc[t-self.timeSpreadPast, 'DateTypeDate'] + dt.timedelta(minutes=(self.timeSpreadPast*15)))
            # print(data.loc[t, 'DateTypeDate'])
            print(f'Some data are missing, skip prediction.')
            print(data[['Date', 'time_close', 'Close', 'DateTypeDate']].tail())
            return
        # enter normal prediction cycle.
        targetIndices = range(t+1-self.timeSpreadPast, t+1)
        top = data.loc[targetIndices, 'High'].max()
        bottom = data.loc[targetIndices, 'Low'].min()
        topBottomArray = nu.linspace(bottom, top, self.resolution)
        graphArray = nu.zeros((self.timeSpreadPast, self.resolution), dtype=nu.int)
        for i, _t in enumerate(targetIndices):
            lowerBound = data.loc[_t, 'Low']
            upperBound = data.loc[_t, 'High']
            graphArray[i, :] = nu.array([ True if lowerBound <= value <= upperBound else False for value in topBottomArray ]) * 1
        # save graphData
        graphData = pd.DataFrame(graphArray, index=data.loc[targetIndices, 'Date'])
        graphName = data.loc[t, 'Date'].split('.')[0].replace('T', '_').replace(':', '-')
        prediction = self.model.predict(graphArray.reshape(1, self.timeSpreadPast, self.resolution, 1))[0]
        # save graph
        if shouldSaveGraph:
            assert graphDataDir != None
            graphData.to_csv(f'{graphDataDir}/{graphName}.csv', index=True, header=True)
            terms = ['+', '=', '-']
            print('@UTC {} Prediction: {} (+:{:.3g}%, =:{:.3g}%, -:{:.3g}%)'.format(now.strftime('%Y-%m-%d %H:%M:%S'), terms[nu.argmax(prediction)], prediction[0]*100, prediction[1]*100, prediction[2]*100))
            fig = pl.figure()
            pl.title('@{} {} (+:{:.3g}%, =:{:.3g}%, -:{:.3g}%)'.format(now.strftime('%Y-%m-%d %H:%M:%S'), terms[nu.argmax(prediction)], prediction[0]*100, prediction[1]*100, prediction[2]*100), fontsize=24)
            pl.xlabel('Date', fontsize=22)
            pl.ylabel('Value', fontsize=22)
            pl.imshow(nu.rot90(graphArray), cmap = 'gray')
            fig.savefig('latestPrediction.png')
            pl.close(fig)
        return int(prediction - 1)


    def __buildModel(self):
        model = kr.models.Sequential([
            kr.layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=(self.timeSpreadPast, self.resolution, 1)),
            kr.layers.MaxPooling2D(pool_size=(4, 4)),
            kr.layers.Conv2D(filters=64, kernel_size=2, activation='relu', input_shape=(self.timeSpreadPast, self.resolution, 1)),
            kr.layers.MaxPooling2D(pool_size=(2, 2)),
            kr.layers.Dropout(0.25),
            kr.layers.Flatten(),
            # kr.layers.Dense(50, activation='relu'),
            kr.layers.Dense(50, activation='tanh'),
            kr.layers.Dropout(0.25),
            kr.layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model


    def __checkAndHandleLoadingModel(self):
        if not self.isModelLoaded:
            if os.path.exists(self.modelPath):
                self.model = kr.models.load_model(self.modelPath)
                self.isModelLoaded = True



if __name__ == '__main__':
    cnnModel = CNNAI()
    worker = PreprocessingWorker(resolution=cnnModel.resolution, timeSpreadPast=cnnModel.timeSpreadPast)

    # worker.processShortermHistoryData(span='15MIN', resolution=cnnModel.resolution, timeSpreadPast=cnnModel.timeSpreadPast)
    cnnModel.train()
    # cnnModel.showModelBreviation(graphDataDir='./StoredData')
