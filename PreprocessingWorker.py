import pandas as pd
import pandas_datareader.data as pdr
import datetime as dt
import numpy as nu
import talib as ta
from matplotlib import pyplot as pl
from coinapi_rest_v1 import CoinAPIv1
import os
import time
from urllib.error import HTTPError


# path = "BTC-JPY.csv"
# coinApiKeys = ["F5F1D416-EAEF-4654-ADA9-D728E9C9A226", "4934D542-652A-46AB-A6A8-BFC8511B251A", "55256919-0DF4-45A0-B899-581C990D6E5E", "94483EBE-A1B9-49DC-8ED8-B35BC986A1A6", "AFA241C6-C5DE-4BD7-8033-4C7175F83A21", "BE9D33EB-F1BB-4435-9614-38EF94DF2C4F"]
coinApiKeys = ["F5F1D416-EAEF-4654-ADA9-D728E9C9A226", "4934D542-652A-46AB-A6A8-BFC8511B251A"]

class PreprocessingWorker():
    def __init__(self, determinateSigma=1.0):
        self.determinateSigma = determinateSigma  # 1.0 * sigma


    # reference: https://note.com/mman/n/n7cccd8bb8961
    def download15MinuteSpanData(self, start, end=dt.datetime.utcnow()):
        key = coinApiKeys.pop()
        client = CoinAPIv1(key)
        # start = dt.datetime(end.year - 3, end.month, end.day, 0, 0, 0)

        dirName = './HistoryData/'
        if not os.path.exists(dirName):
            os.mkdir(dirName)

        date = start
        while date <= end:
            filePath = dirName + "/" + date.strftime('%Y_%m_%d_%H_%M') + '.csv'
            _start = date
            _end = _start + dt.timedelta(days=1) + dt.timedelta(seconds=30)
            if _end > end:
                _end = end + dt.timedelta(seconds=30)

            response = None
            try:
                response = client.ohlcv_historical_data('BITFLYER_SPOT_BTC_JPY', {'period_id': '15MIN', 'time_start': _start.isoformat(), 'time_end': _end.isoformat()})
            except HTTPError as error:
                if error.code == 429:
                    if len(coinApiKeys) == 0:
                        break
                    print(f'Key: {key} is used out for today.')
                    key = coinApiKeys.pop()
                    print(f'Trying new key: {key} ...')
                    client = CoinAPIv1(key)
                    response = client.ohlcv_historical_data('BITFLYER_SPOT_BTC_JPY', {'period_id': '15MIN', 'time_start': _start.isoformat(), 'time_end': _end.isoformat()})

            # if data of the specific min exists, update it.
            if os.path.exists(filePath):
                data = pd.read_csv(filePath)
                responseData = pd.DataFrame(response)
                responseData = responseData.rename(columns={
                    'price_open': 'Open',
                    'price_close': 'Close',
                    'price_high': 'High',
                    'price_low': 'Low',
                    'volume_traded': 'Volume',
                    'time_period_start': 'Date'
                })
                data = pd.concat([data, responseData])
                del responseData
            # if data of the date still not exists, create from the response.
            else:
                data = pd.DataFrame(response)
                data = data.rename(columns={
                    'price_open': 'Open',
                    'price_close': 'Close',
                    'price_high': 'High',
                    'price_low': 'Low',
                    'volume_traded': 'Volume',
                    'time_period_start': 'Date'
                })
            # save data
            data.to_csv(filePath, index=False)
            # prepare for next loop
            date += dt.timedelta(days=1)
            print(f'Date {_start} to {_end} completed.')
            time.sleep(5)


    def processShortermHistoryData(self, span='15MIN', shouldShowData=True):
        print('Start preprocessing history data ...')
        # general constants
        dirName = './HistoryData'
        fileNames = filter(lambda name: '.csv' in name, os.listdir(dirName))
        fileNames = sorted([ (name, dt.datetime.strptime(name.split('.csv')[0], '%Y_%m_%d_%H_%M')) for name in fileNames ], key=lambda pair: pair[1])
        storedDirName = './LabeledData'
        storedFilePath = f'{storedDirName}/{span}.csv'
        # switch for span
        if span == '15MIN':
            # fetch data
            name, date = fileNames[0]
            data = pd.read_csv(f'{dirName}/{name}')
            for name, date in fileNames[1:]:
                _newData = pd.read_csv(f'{dirName}/{name}')
                _newData = _newData.drop(_newData.index[[-1]])
                data = pd.concat([data, _newData])
            data = data.drop(['Volume', 'trades_count'], axis=1).dropna().reset_index(drop=True)
            del fileNames
            self.__processData(data, storedFilePath, shouldShowData=True)


    def __processData(self, data, path, zoneLength=20, shouldShowData=True):

        # fetch data
        # data = pd.read_csv(path).dropna().reset_index(drop=True)
        # _end = dt.date.today()
        # _start = dt.date(_end.year - 4, _end.month, _end.day)
        # data = pdr.DataReader('BTC-JPY', 'yahoo', _start, _end).dropna().reset_index(drop=False)

        # calculate labels
        trainSamples = data['Close'].values[:-zoneLength]
        trainLabels = []
        for index in data.index.values[:-zoneLength]:
            focusingValue = data.iloc[index].loc['Close']
            targetIndices = range(index, index+zoneLength)
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
        _nas = nu.zeros(zoneLength)
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
        data['BB+2sigma'], data['BBmiddle'], data['BB-2sigma'] = ta.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        data['Sigma'] = (data['BB+2sigma'] - data['BBmiddle'])/2
        data['BBPosition'] = (data['Close'] - data['BBmiddle'])/data['Sigma']

        data.to_csv(path, index=False, header=True)

        if shouldShowData:
            self.showData(path, data)

        print('Ends preprocessing history data.')
        return data


    def showData(self, path, data=None):
        if data:
            pass
        else:
            data = pd.read_csv(path).dropna().reset_index(drop=False)

        _0samples = data[data.ClassLabel == 0][['BBPosition', 'RSI14']]
        _1samples = data[data.ClassLabel == 1][['BBPosition', 'RSI14']]
        _2samples = data[data.ClassLabel == 2][['BBPosition', 'RSI14']]
        pl.scatter(_0samples['BBPosition'], _0samples['RSI14'], c='g')
        pl.scatter(_1samples['BBPosition'], _1samples['RSI14'], c='gray', alpha=0.3)
        pl.scatter(_2samples['BBPosition'], _2samples['RSI14'], c='r')
        pl.show()


    def dampShortermDataIntoSpanData(self, span='15MIN', end=dt.datetime.utcnow()):
        dirName = './StoredData'
        fileNames = filter(lambda name: '.csv' in name, os.listdir(dirName))
        fileNames = sorted([ (name, dt.datetime.strptime(name.split('.csv')[0], '%Y_%m_%d')) for name in fileNames ], key=lambda pair: pair[1])
        storedDirName = './LabeledData'
        storedFilePath = f'{storedDirName}/{span}.csv'

        data = pd.read_csv(storedFilePath)
        latestDate = dt.datetime.fromisoformat(data.tail(1)['Date'].split('.')[0])

        if span == '15MIN':
            date = latestDate
            while date + dt.timedelta(minutes=15) <= end:
                _start = dt.datetime(date.year, date.month, date.day, date.hour, (int(date.minute)//15)*15, date.second)
                _end = _start += dt.timedelta(minutes=15)
                pathForDate = f'{dirName}/' + date.strftime('%Y_%m_%d') + '.csv'
                if pathForDate in fileNames:
                    rawData = pd.read_csv(pathForDate).dropna()
                    rawData['Date'] = nu.array([dt.datetime.fromisoformat(str.split('.')[0]) for str in rawData['timestamp']])
                    _targets = rawData[rawData.Date >= _start][rawData.Date <= _end]['ltp'].values
                    _open = _targets[0]
                    _close = _targets[-1]
                    _high = _targets.max()
                    _low = _targets.min()
                    newData = pd.DateFrame({
                        'Date': date.isoformat() + '0Z',
                        'time_period_end': (date + dt.timedelta(minutes=15)).isoformat() + '0Z',
                        'Open': _open,
                        'Close': _close,
                        'High': _high,
                        'Low': _low
                    })
                    newData['time_open'] = newData['Date']
                    newData['time_close'] = newData['time_period_end']
                    data = pd.concat([data, newData])
                else:
                    continue
        data.to_csv(storedFilePath, index=False, header=True)




if __name__ == '__main__':
    worker = PreprocessingWorker()
    # worker.process()
    # worker.showData(path='./LabeledData/15MIN.csv')

    # start = dt.datetime(2018, 6, 28, 0, 0, 0)
    # now = dt.datetime.utcnow()
    # worker.download15MinuteSpanData(start=start, end=now)

    worker.dampShortermDataIntoSpanData()
    
    # worker.processShortermHistoryData(span='15MIN')
