import pandas as pd
import datetime as dt
import numpy as nu
from matplotlib import pyplot as pl
from coinapi_rest_v1 import CoinAPIv1
import os
import time
from urllib.error import HTTPError
import multiprocessing as mp


# path = "BTC-JPY.csv"
coinApiKeys = ["BF49F16C-E6CF-4B26-A22E-F32599C6E404", "F5F1D416-EAEF-4654-ADA9-D728E9C9A226", "4934D542-652A-46AB-A6A8-BFC8511B251A", "55256919-0DF4-45A0-B899-581C990D6E5E", "94483EBE-A1B9-49DC-8ED8-B35BC986A1A6", "AFA241C6-C5DE-4BD7-8033-4C7175F83A21", "BE9D33EB-F1BB-4435-9614-38EF94DF2C4F"]


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


    def processShortermHistoryData(self, span='15MIN', shouldShowData=True, resolution=int(24*4), coreAmount=1):
        _start = dt.datetime.now()
        print('Start preprocessing history data ...')
        # general constants
        dirName = './HistoryData'
        fileNames = filter(lambda name: '.csv' in name, os.listdir(dirName))
        fileNames = sorted([ (name, dt.datetime.strptime(name.split('.csv')[0], '%Y_%m_%d_%H_%M')) for name in fileNames ], key=lambda pair: pair[1])
        storedDirName = './LabeledData'
        storedFilePath = f'{storedDirName}/{span}.csv'
        graphDataDir = f'./{storedDirName}/graphData'
        if not os.path.exists(graphDataDir):
            os.mkdir(graphDataDir)
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
            data['LabelCNNPost1'] = nu.nan
            del fileNames
            # processTank = []
            # tAmountPerTime = int(data.index.values[int(24*4):-27].ravel().shape[0] / coreAmount)
            # ts = None
            # labelQueue = mp.SimpleQueue()
            # for core in range(coreAmount):
            #     if core != coreAmount-1:
            #         ts = data.index.values[int(24*4 + core*tAmountPerTime):int(24*4 + (core+1)*tAmountPerTime)]
            #     else:
            #         ts = data.index.values[int(24*4 + core*tAmountPerTime):-27]
            #     process = mp.Process(target=generateGraphData, args=(data, ts, resolution, graphDataDir, labelQueue))
            #     process.start()
            #     processTank.append(process)
            # for process in processTank:
            #     process.join()
            # calculate graphData and update label
            for t in data.index.values[int(24*4):-27]:
                targetIndices = range(t-24*4 +1, t+1 +1)
                # highs = data.loc[targetIndices, 'High'].values.ravel()
                # lows = data.loc[targetIndices, 'Low'].values.ravel()
                top = data.loc[targetIndices, 'High'].max()
                down = data.loc[targetIndices, 'Low'].min()
                topDownArray = nu.linspace(down, top, resolution)
                # find label
                nowMiddle = (data.loc[t, 'High'] + data.loc[t, 'Low'])/2
                futureMiddle = (data.loc[t+1:t+25, 'High'].values.ravel().mean() + data.loc[t+1:t+25, 'Low'].values.ravel().mean())/2
                sigma = nu.abs((data.loc[t, 'High'] - data.loc[t, 'Low'])/(2.57*2))
                if futureMiddle >= nowMiddle + 0.84*sigma:
                    data.loc[t, 'LabelCNNPost1'] = 0
                elif futureMiddle >= nowMiddle - 0.84*sigma:
                    data.loc[t, 'LabelCNNPost1'] = 1
                else:
                    data.loc[t, 'LabelCNNPost1'] = 2

                graphArray = nu.zeros((24*4, resolution), dtype=nu.int)
                for i, _t in enumerate(targetIndices[:-1]):
                    lowerBound = data.loc[_t, 'Low']
                    upperBound = data.loc[_t, 'High']
                    graphArray[i, :] = nu.array([ True if lowerBound <= value <= upperBound else False for value in topDownArray ]) * 1
                graphData = pd.DataFrame(graphArray, index=data.loc[targetIndices[:-1], 'Date'])
                graphName = data.loc[t, 'Date'].split('.')[0].replace('T', '_').replace(':', '-')
                graphData.to_csv(f'{graphDataDir}/{graphName}.csv', index=True, header=True)
            # while not labelQueue.empty():
            #     index, label = labelQueue.get()
            #     data.loc[index, 'LabelCNNPost1'] = label
            data.to_csv(storedFilePath, index=False, header=True)

        timeCost = (dt.datetime.now() - _start).total_seconds()
        print(f'History data preprocessing done with time cost: {timeCost} seconds')


    def show(self, graphData, label):
        pl.title(label)
        pl.imshow(graphData)
        pl.show()
        time.sleep(3)


    def dumpShortermDataIntoSpanData(self, span='15MIN', end=dt.datetime.utcnow()):
        dirName = './StoredData'
        fileNames = filter(lambda name: '.csv' in name, os.listdir(dirName))
        fileNames = sorted([ (name, dt.datetime.strptime(name.split('.csv')[0], '%Y_%m_%d')) for name in fileNames ], key=lambda pair: pair[1])
        fileNames = [ pair[0] for pair in fileNames ]
        storedDirName = './LabeledData'
        storedFilePath = f'{storedDirName}/{span}.csv'

        data = pd.read_csv(storedFilePath)
        latestDate = dt.datetime.fromisoformat(data.tail(1)['Date'].values[0].split('.')[0])

        if span == '15MIN':
            date = latestDate
            while date + dt.timedelta(minutes=15) <= end:
                _start = dt.datetime(date.year, date.month, date.day, date.hour, date.minute, 0)
                _end = _start + dt.timedelta(minutes=15)
                fileName = date.strftime('%Y_%m_%d') + '.csv'
                if fileName in fileNames:
                    pathForDate = f'{dirName}/{fileName}'
                    print(f'{pathForDate} in files, start damping ...')
                    rawData = pd.read_csv(pathForDate).dropna()
                    rawData['Date'] = nu.array([dt.datetime.fromisoformat(str.split('.')[0]) for str in rawData['timestamp']])
                    _targets = rawData[rawData.Date >= _start][rawData.Date <= _end]['ltp'].values
                    if _targets.shape[0] > 0:
                        _open = _targets[0]
                        _close = _targets[-1]
                        _high = _targets.max()
                        _low = _targets.min()
                        newData = pd.DataFrame({
                            'Date': date.isoformat() + '.0000000Z',
                            'time_period_end': (date + dt.timedelta(minutes=15)).isoformat() + '.0000000Z',
                            'Open': _open,
                            'Close': _close,
                            'High': _high,
                            'Low': _low
                        }, index=[data.index.values[-1]+1])
                        newData['time_open'] = newData['Date']
                        newData['time_close'] = newData['time_period_end']
                        data = pd.concat([data, newData])
                else:
                    print(f'{fileName} not in files, continue ...')
                # prepare for next loop
                date = _end
        data = data.reset_index(drop=True)
        updatedData = self.__processData(data.tail(100), storedFilePath, shouldShowData=False)
        data.iloc[-50:] = updatedData.tail(50)
        data.reset_index(drop=True).to_csv(storedFilePath, index=False, header=True)
        return data


def generateGraphData(data, ts, resolution, graphDataDir, queue):
    # calculate graphData and update label
    for t in ts:
        targetIndices = range(t-24*4 +1, t+1 +1)
        # highs = data.loc[targetIndices, 'High'].values.ravel()
        # lows = data.loc[targetIndices, 'Low'].values.ravel()
        top = data.loc[targetIndices, 'High'].max()
        down = data.loc[targetIndices, 'Low'].min()
        topDownArray = nu.linspace(down, top, resolution)
        # find label
        nowMiddle = (data.loc[t, 'High'] + data.loc[t, 'Low'])/2
        futureMiddle = (data.loc[t+1:t+25, 'High'].values.ravel().mean() + data.loc[t+1:t+25, 'Low'].values.ravel().mean())/2
        sigma = nu.abs((data.loc[t, 'High'] - data.loc[t, 'Low'])/(2.57*2))
        if futureMiddle >= nowMiddle + 0.84*sigma:
            queue.put((t, 0))
        elif futureMiddle >= nowMiddle - 0.84*sigma:
            queue.put((t, 1))
        else:
            queue.put((t, 2))
            # data.loc[t, 'LabelCNNPost1'] = 2

        graphArray = nu.zeros((24*4, resolution), dtype=nu.int)
        for i, _t in enumerate(targetIndices[:-1]):
            lowerBound = data.loc[_t, 'Low']
            upperBound = data.loc[_t, 'High']
            graphArray[i, :] = nu.array([ True if lowerBound <= value <= upperBound else False for value in topDownArray ]) * 1
        graphData = pd.DataFrame(graphArray, index=data.loc[targetIndices[:-1], 'Date'])
        graphName = data.loc[t, 'Date'].split('.')[0].replace('T', '_').replace(':', '-')
        graphData.to_csv(f'{graphDataDir}/{graphName}.csv', index=True, header=True)


if __name__ == '__main__':
    worker = PreprocessingWorker()
    # worker.process()
    # worker.showData(path='./LabeledData/15MIN.csv')

    # start = dt.datetime(2020,  7, 15, 0, 0, 0)
    # now = dt.datetime.utcnow()
    # worker.download15MinuteSpanData(start=start, end=now)

    # worker.dumpShortermDataIntoSpanData()

    worker.processShortermHistoryData(span='15MIN')
