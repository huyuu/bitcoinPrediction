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


path = "BTC-JPY.csv"
# coinApiKeys = ["55256919-0DF4-45A0-B899-581C990D6E5E", "94483EBE-A1B9-49DC-8ED8-B35BC986A1A6", "AFA241C6-C5DE-4BD7-8033-4C7175F83A21", "BE9D33EB-F1BB-4435-9614-38EF94DF2C4F"]
coinApiKeys = ["F5F1D416-EAEF-4654-ADA9-D728E9C9A226", "4934D542-652A-46AB-A6A8-BFC8511B251A"]

class PreprocessingWorker():
    def __init__(self, zoneLength=30, determinateSigma=1.0):
        self.zoneLength = zoneLength  # days
        self.determinateSigma = determinateSigma  # 1.0 * sigma


    # reference: https://note.com/mman/n/n7cccd8bb8961
    def downloadMinuteSpanData(self, start, end=dt.datetime.utcnow()):
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
                # responseData = pd.DataFrame(client.ohlcv_historical_data('COINCHECK_SPOT_BTC_JPY', {'period_id': '15MIN', 'time_start': _start.isoformat(), 'limit': 8640}))
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
                # data = pd.DataFrame(client.ohlcv_historical_data('COINCHECK_SPOT_BTC_JPY', {'period_id': '15MIN', 'time_start': _start.isoformat(), 'limit': 8640}))
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
            time.sleep(10)


    def processHistoryData(self, shouldShowData=True):
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
    # worker.showData()
    start = dt.datetime(2017, 12, 9, 0, 0, 0)
    now = dt.datetime.utcnow()
    worker.downloadMinuteSpanData(start=start, end=now)
