import pandas as pd
import pandas_datareader.data as pdr
import datetime as dt
import numpy as nu
import talib as ta
from matplotlib import pyplot as pl

path = "BTC-JPY.csv"


class PreprocessingWorker():
    def __init__(self, zoneLength=30, determinateSigma=1.0):
        self.zoneLength = zoneLength  # days
        self.determinateSigma = determinateSigma  # 1.0 * sigma


    def process(self, shouldShowData=True):
        print('Start preprocessing history data ...')
        # fetch data
        # data = pd.read_csv(path).dropna().reset_index(drop=True)
        _end = dt.date.today()
        _start = dt.date(_end.year - 4, _end.month, _end.day)
        data = pdr.DataReader('BTC-JPY', 'yahoo', _start, _end).dropna().reset_index(drop=False)
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
        _nas = nu.zeros(self.zoneLength)
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
        # calculate RSI30
        data['RSI30'] = ta.RSI(data['Close'], timeperiod=30)
        # calculate BB
        # reference: https://note.com/10mohi6/n/n92c6ed9af759
        data['BB+1sigma'], data['BBmiddle'], data['BB-1sigma'] = ta.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        data['Sigma'] = (data['BB+1sigma'] - data['BBmiddle'])/2
        data['BBPosition'] = (data['Close'] - data['BBmiddle'])/data['Sigma']

        data.to_csv(path, index=False, header=True)

        if shouldShowData:
            self.showData()

        print('Ends preprocessing history data.')
        return data


    def showData(self, data=None):
        if data == None:
            data = pd.read_csv(path).dropna().reset_index(drop=False)

        _0samples = data[data.ClassLabel == 0][['BBPosition', 'RSI14']]
        _1samples = data[data.ClassLabel == 1][['BBPosition', 'RSI14']]
        _2samples = data[data.ClassLabel == 2][['BBPosition', 'RSI14']]
        pl.scatter(_0samples['BBPosition'], _0samples['RSI14'], c='g')
        pl.scatter(_1samples['BBPosition'], _1samples['RSI14'], c='gray', alpha=0.3)
        pl.scatter(_2samples['BBPosition'], _2samples['RSI14'], c='r')
        pl.show()


if __name__ == '__main__':
    worker = PreprocessingWorker()
    # worker.process()
    worker.showData()
