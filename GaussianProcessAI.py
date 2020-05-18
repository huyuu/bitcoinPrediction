import numpy as nu
import pandas as pd
import sklearn as sk
from sklearn import gaussian_process as gp
import pickle
from matplotlib import pyplot as pl
import os
# import talib as ta

from plotGraph import plot

modelFileName = "gpmodel.sav"
path = "BTC-JPY.csv"


class GaussianProcessAI():
    def __init__(self, zoneLength=30, determinateSigma=1.0):
        self.zoneLength = zoneLength  # days
        self.determinateSigma = determinateSigma  # 1.5 * sigma
        longtermKernel = 1.0 * gp.kernels.RBF(5.0)
        shorttermKernel = 5.0 * gp.kernels.RationalQuadratic(length_scale=0.1, alpha=0.78)
        noiseKernel = 0.1 * gp.kernels.WhiteKernel(1e-2)
        kernel = longtermKernel + shorttermKernel + noiseKernel
        self.model = gp.GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=9)


    def varifyPrediction(self, testifyRadio=0.2):
        if os.path.exists(modelFileName):
            self.model = pickle.load(open(modelFileName, 'rb'))
        else:
            self.trainFromHistoryData(testifyRadio)

        data = pd.read_csv(path).dropna()
        data = data.head(data.index.values.shape[0] - self.zoneLength)
        data['Delta'] = data['Close'] - data['Open']
        amount = int(data.index.values.shape[0] * (1-testifyRadio))
        samples = data['Delta'].values[amount:].reshape(-1, 1)
        correctLabels = data['ClassLabel'].values[amount:].reshape(-1, 1)
        predictedLabels = self.model.predict(samples)
        pl.scatter(samples, predictedLabels, c='r')
        pl.scatter(samples, correctLabels.ravel(), alpha=0.5)
        pl.show()


    def preprocessData(self):
        print('Start preprocessing history data ...')
        # fetch data
        data = pd.read_csv(path).dropna().reset_index(drop=True)
        # calculate labels
        trainSamples = data['Close'].values[:-self.zoneLength]
        trainLabels = []
        for index in data.index.values[:-self.zoneLength]:
            focusingValue = data.iloc[index].loc['Close']
            targetIndices = range(index, index+self.zoneLength)
            points = nu.array([])
            for targetIndex in targetIndices:
                targetData = data.iloc[targetIndex][['High', 'Low']]
                targetPoints = (targetData['High'] - targetData['Low']) * nu.random.rand(50) + targetData['Low']
                points = nu.concatenate([points, targetPoints])
            std = nu.std(points)
            mean = nu.mean(points)
            if focusingValue >= mean + self.determinateSigma * std:
                label = 2
            elif focusingValue <= mean - self.determinateSigma * std:
                label = 0
            else:
                label = 1
            trainLabels.append(label)
        trainLabels = nu.array(trainLabels, dtype=nu.int)
        _nas = nu.zeros(self.zoneLength, dtype=nu.int)
        _nas[:] = nu.nan
        data['ClassLabel'] = nu.concatenate([trainLabels, _nas])
        # calculate Delta
        data['Delta'] = data['Close'] - data['Open']
        # calculate RSI14
        rsi14 = []
        for index in data.index.values[14:]:
            targetIndices = range(index-14, index)
            up = 0
            down = 0
            for targetIndex in targetIndices:
                delta = data.iloc[targetIndex]['Delta']
                if delta >= 0:
                    up += delta
                else:
                    down += nu.abs(delta)
            rsi14.append(up/(up+down))
        rsi14 = nu.array(rsi14)
        _nas = nu.zeros(14)
        _nas[:] = nu.nan
        data['RSI14'] = nu.concatenate([_nas, rsi14])
        # calculate BB
        # reference: https://note.com/10mohi6/n/n92c6ed9af759
        data['BB+1sigma'], data['BBmiddle'], data['BB-1sigma'] = ta.BBANDS(data['Close'], timeperiod=30)
        data['Sigma'] = data['BB+1sigma'] - data['BBmiddle']
        data['BBPosition'] = (data['Close'] - data['BBmiddle'])/data['Sigma']

        data.to_csv(path, index=False, header=True)
        print('Ends preprocessing history data.')
        return data


    def trainFromHistoryData(self, testifyRadio=0.2, shouldPreprocessData=False):
        # fetch data
        data = pd.read_csv(path).dropna()
        if not 'ClassLabel' in data.columns or shouldPreprocessData:
            data = self.preprocessData()
        data = data.head(data.index.values.shape[0] - self.zoneLength)
        # set samples
        testSamplesAmount = int(data.index.values.shape[0] * testifyRadio)
        trainSamplesAmount = data.index.values.shape[0] - testSamplesAmount
        trainSamples = data[['Delta', 'RSI14']].values[:trainSamplesAmount, :]
        trainLabels = data['ClassLabel'].values[:trainSamplesAmount].reshape(-1, 1)
        testSamples = data[['Delta', 'RSI14']].values[trainSamplesAmount:, :]
        testLabels = data['ClassLabel'].values[trainSamplesAmount:].reshape(-1, 1)

        _0samples = data[data.ClassLabel == 0][['BBPosition', 'RSI14']]
        _1samples = data[data.ClassLabel == 1][['BBPosition', 'RSI14']]
        _2samples = data[data.ClassLabel == 2][['BBPosition', 'RSI14']]
        # pl.scatter(_0samples['BBPosition'], _0samples['RSI14'], c='g')
        # pl.scatter(_1samples['BBPosition'], _1samples['RSI14'], c='gray', alpha=0.3)
        # pl.scatter(_2samples['BBPosition'], _2samples['RSI14'], c='r')
        # pl.show()
        # start training
        self.model.fit(trainSamples, trainLabels)
        print(f'Model: {self.model.kernel_}')
        print(f'Log_Marginal_Likelihood: {self.model.log_marginal_likelihood(self.model.kernel_.theta)}')
        pickle.dump(self.model, open(modelFileName, 'wb'))
        score = self.model.score(testSamples, testLabels)
        print(score)



if __name__ == '__main__':
    ai = GaussianProcessAI()
    # ai.preprocessData()
    ai.trainFromHistoryData()

    # ai.varifyPrediction()
    # plot()
