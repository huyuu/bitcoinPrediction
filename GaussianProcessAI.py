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

from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from plotGraph import plot

modelFileName = "gpmodel.sav"
path = "BTC-JPY.csv"


class GaussianProcessAI():
    def __init__(self, zoneLength=30, determinateSigma=1.0):
        self.zoneLength = zoneLength  # days
        self.determinateSigma = determinateSigma  # 1.5 * sigma
        # longtermKernel = 1.0 * gp.kernels.RBF(5.0)
        # shorttermKernel = 5.0 * gp.kernels.RationalQuadratic(length_scale=0.1, alpha=0.78)
        # noiseKernel = 0.1 * gp.kernels.WhiteKernel(1e-2)
        # kernel = longtermKernel + shorttermKernel + noiseKernel
        # self.model = gp.GaussianProcessClassifier(kernel=1.0*gp.kernels.RBF(1.0), n_restarts_optimizer=9)
        self.models = {
            'GPC': gp.GaussianProcessClassifier(kernel=1.0*gp.kernels.RBF(1.0), n_restarts_optimizer=9),
            'KNC': KNeighborsClassifier(3),
            'SVC': SVC(gamma=2, C=1),
            'RFC': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            'MLP': MLPClassifier(alpha=1, max_iter=1000),
            'ABC': AdaBoostClassifier(),
            'GNB': GaussianNB(),
            'QDA': QuadraticDiscriminantAnalysis()
        }


    def varifyPrediction(self):
        if os.path.exists(modelFileName):
            self.model = pickle.load(open(modelFileName, 'rb'))
        else:
            self.trainFromHistoryData()
        _BBsamples = nu.linspace(0, 1, 20)
        _RSIsamples = nu.linspace(-2, 2, 20)
        _BBsamplesGrid, _RSIsamplesGrid = nu.meshgrid(_BBsamples, _RSIsamples, indexing='ij')
        samples = nu.concatenate([_BBsamplesGrid.reshape(-1, 1), _RSIsamplesGrid.reshape(-1, 1)], axis=1)

        probabilities = self.model.predict_proba(samples)
        predictedLabels = self.model.predict(samples)
        data = pd.DataFrame(nu.concatenate([samples, predictedLabels.reshape(-1, 1), probabilities[:, 0].reshape(-1, 1), probabilities[:, 1].reshape(-1, 1), probabilities[:, 2].reshape(-1, 1)], axis=1), columns=['BBPosition', 'RSI14', 'ClassLabel', 'Label0_prob', 'Label1_prob', 'Label2_prob'])
        print(data)
        _0samples = data[data.ClassLabel == 0]
        _1samples = data[data.ClassLabel == 1]
        _2samples = data[data.ClassLabel == 2]
        pl.scatter(_0samples['BBPosition'], _0samples['RSI14'], c='g')
        pl.scatter(_1samples['BBPosition'], _1samples['RSI14'], c='gray', alpha=0.3)
        pl.scatter(_2samples['BBPosition'], _2samples['RSI14'], c='r')
        pl.show()

        ax = pl.axes(projection='3d')
        ax.plot_surface(_BBsamplesGrid, _RSIsamplesGrid, data['Label0_prob'].values.reshape(20, 20), cmap='Blues')
        pl.show()

        fig = pl.figure()
        ax = pl.axes(projection='3d')
        ax.plot_surface(_BBsamplesGrid, _RSIsamplesGrid, data['Label1_prob'].values.reshape(20, 20), cmap='gray')
        pl.show()

        fig = pl.figure()
        ax = pl.axes(projection='3d')
        ax.plot_surface(_BBsamplesGrid, _RSIsamplesGrid, data['Label2_prob'].values.reshape(20, 20), cmap='Reds')
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
        data = pd.read_csv(path)
        if not 'ClassLabel' in data.columns or shouldPreprocessData:
            data = self.preprocessData()
        data = data.dropna().reset_index(drop=True)
        # set samples
        testSamplesAmount = int(data.index.values.shape[0] * testifyRadio)
        trainSamplesAmount = data.index.values.shape[0] - testSamplesAmount
        trainSamples = data[['BBPosition', 'RSI14']].values[:trainSamplesAmount, :]
        trainLabels = data['ClassLabel'].values[:trainSamplesAmount].reshape(-1, 1)
        testSamples = data[['BBPosition', 'RSI14']].values[trainSamplesAmount:, :]
        testLabels = data['ClassLabel'].values[trainSamplesAmount:].reshape(-1, 1)

        # _0samples = data[data.ClassLabel == 0][['BBPosition', 'RSI14']]
        # _1samples = data[data.ClassLabel == 1][['BBPosition', 'RSI14']]
        # _2samples = data[data.ClassLabel == 2][['BBPosition', 'RSI14']]
        # pl.scatter(_0samples['BBPosition'], _0samples['RSI14'], c='g')
        # pl.scatter(_1samples['BBPosition'], _1samples['RSI14'], c='gray', alpha=0.3)
        # pl.scatter(_2samples['BBPosition'], _2samples['RSI14'], c='r')
        # pl.show()
        # start training
        print('Start training model ...')
        _start = dt.datetime.now()
        # self.model.fit(trainSamples, trainLabels.ravel())
        # print(f'Model: {self.model.kernel_}')
        # print(f'Log_Marginal_Likelihood: {self.model.log_marginal_likelihood(self.model.kernel_.theta)}')
        # pickle.dump(self.model, open(modelFileName, 'wb'))
        # score = self.model.score(testSamples, testLabels)
        # print(score)

        processes = []
        for name, model in self.models.items():
            process = mp.Process(target=trainModel, args=(name, model, trainSamples, trainLabels, testSamples, testLabels))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        timeCost = (dt.datetime.now() - _start).total_seconds()
        print(f'Model training ends. (time cost: {timeCost} seconds)')


def trainModel(name, model, trainSamples, trainLabels, testSamples, testLabels):
    model.fit(trainSamples, trainLabels.ravel())
    score = model.score(testSamples, testLabels.ravel())
    print(f'{name} model score: {score}')


if __name__ == '__main__':
    ai = GaussianProcessAI()
    # ai.preprocessData()
    ai.trainFromHistoryData()
    # ai.varifyPrediction()
    # plot()
