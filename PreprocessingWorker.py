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
    def __init__(self, resolution, timeSpreadPast):
        self.resolution = resolution
        self.timeSpreadPast = timeSpreadPast


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
            # if BITFLYER data not available, try BITBANK
            if type(response) is list and len(response) <= self.timeSpreadPast/2:
                try:
                    response = client.ohlcv_historical_data('BITBANK_SPOT_BTC_JPY', {'period_id': '15MIN', 'time_start': _start.isoformat(), 'time_end': _end.isoformat()})
                except HTTPError as error:
                    if error.code == 429:
                        if len(coinApiKeys) == 0:
                            break
                        print(f'Key: {key} is used out for today.')
                        key = coinApiKeys.pop()
                        print(f'Trying new key: {key} ...')
                        client = CoinAPIv1(key)
                        response = client.ohlcv_historical_data('BITBANK_SPOT_BTC_JPY', {'period_id': '15MIN', 'time_start': _start.isoformat(), 'time_end': _end.isoformat()})
            # # if data of the specific min exists, update it.
            # if os.path.exists(filePath):
            #     data = pd.read_csv(filePath)
            #     responseData = pd.DataFrame(response)
            #     responseData = responseData.rename(columns={
            #         'price_open': 'Open',
            #         'price_close': 'Close',
            #         'price_high': 'High',
            #         'price_low': 'Low',
            #         'volume_traded': 'Volume',
            #         'time_period_start': 'Date'
            #     })
            #     data = pd.concat([data, responseData])
            #     del responseData
            # # if data of the date still not exists, create from the response.
            # else:
            #     data = pd.DataFrame(response)
            #     data = data.rename(columns={
            #         'price_open': 'Open',
            #         'price_close': 'Close',
            #         'price_high': 'High',
            #         'price_low': 'Low',
            #         'volume_traded': 'Volume',
            #         'time_period_start': 'Date'
            #     })
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


    def processShortermHistoryData(self, resolution, timeSpreadPast, span='15MIN', startDate=None, shouldShowData=True):
        _start = dt.datetime.now()
        timeSpreadFuture = int(timeSpreadPast / 4)
        print('Start preprocessing history data ...')
        # general constants
        dirName = './HistoryData'
        storedDirName = './LabeledData'
        storedFilePath = f'{storedDirName}/{span}.csv'
        graphDataDir = f'./{storedDirName}/graphData'
        if not os.path.exists(graphDataDir):
            os.mkdir(graphDataDir)
        # switch for span
        if span == '15MIN':
            if startDate != None and os.path.exists(storedFilePath):
                fileNames = filter(lambda name: '.csv' in name and os.stat(f'{dirName}/{name}').st_size > 10, os.listdir(dirName))
                fileNames = [ (name, dt.datetime.strptime(name.split('.csv')[0], '%Y_%m_%d_%H_%M')) for name in fileNames ]
                fileNames = filter(lambda pair: pair[1] >= startDate, fileNames)
                fileNames = sorted(fileNames, key=lambda pair: pair[1])
                # fetch old data
                data = pd.read_csv(storedFilePath)
                data['DateTypeDate'] = stringToDate(data['Date'].values.ravel())
                # fetch new data
                ts = nu.array([], dtype=nu.int)
                for name, date in fileNames:
                    newData = pd.read_csv(f'{dirName}/{name}')
                    # drop last row
                    newData = newData.drop(newData.index[[-1]])
                    newData['DateTypeDate'] = stringToDate(newData['Date'].values.ravel())
                    rowIndex = data.loc[data['Date'] == dateToString(date)].index.values
                    if rowIndex.shape[0] != 0:
                        newIndices = range(rowIndex[0], rowIndex[0] + newData.index.values.shape[0])
                        for i, row in enumerate(newIndices):
                            data.loc[row] = newData.loc[i]
                        ts = nu.concatenate([ts, newIndices])
                    else: # add to the last line
                        newIndices = range(data.index.values.shape[0], data.index.values.shape[0] + newData.index.values.shape[0])
                        for i, row in enumerate(newIndices):
                            data.loc[row] = newData.loc[i]
                        ts = nu.concatenate([ts, newIndices])
                        data = data.sort_values('DateTypeDate').reset_index(drop=True)
                # calculate graphData and label
                data = generateGraphDataAndLabel(data=data, ts=ts, resolution=resolution, timeSpreadPast=int(timeSpreadPast), timeSpreadFuture=int(timeSpreadFuture), graphDataDir=graphDataDir)
                del data['DateTypeDate']
                data.to_csv(storedFilePath, index=False, header=True)

            else: # don't have startDate
                fileNames = filter(lambda name: '.csv' in name and os.stat(f'{dirName}/{name}').st_size > 10, os.listdir(dirName))
                fileNames = [ (name, dt.datetime.strptime(name.split('.csv')[0], '%Y_%m_%d_%H_%M')) for name in fileNames ]
                fileNames = sorted(fileNames, key=lambda pair: pair[1])
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
                # calculate graphData and update label
                data = generateGraphDataAndLabel(data=data, ts=data.index.values.ravel(), resolution=resolution, timeSpreadPast=int(timeSpreadPast), timeSpreadFuture=int(timeSpreadFuture), graphDataDir=graphDataDir)
                # for t in data.index.values[timeSpreadPast:-timeSpreadFuture-2]:
                #     if t-timeSpreadPast+1 in data.index.values:
                #         if t+timeSpreadFuture in data.index.values:
                #             # prepare for labeling
                #             targetIndices = range(t-timeSpreadPast, t+1)
                #             top = data.loc[targetIndices, 'High'].max()
                #             down = data.loc[targetIndices, 'Low'].min()
                #             topDownArray = nu.linspace(down, top, resolution)
                #             # find label
                #             nowMiddle = (data.loc[t, 'High'] + data.loc[t, 'Low'])/2
                #             futureMiddle = (data.loc[t+1:t+timeSpreadFuture+1, 'High'].values.ravel().mean() + data.loc[t+1:t+timeSpreadFuture+1, 'Low'].values.ravel().mean())/2
                #             sigma = nu.abs((data.loc[t, 'High'] - data.loc[t, 'Low'])/(1.96*2))
                #             if futureMiddle >= nowMiddle + 1.0*sigma:
                #                 data.loc[t, 'LabelCNNPost1'] = 0
                #             elif futureMiddle >= nowMiddle - 1.0*sigma:
                #                 data.loc[t, 'LabelCNNPost1'] = 1
                #             else:
                #                 data.loc[t, 'LabelCNNPost1'] = 2
                #         else: # latest data which is unable to be labeled
                #
                #         # draw graph:
                #         graphArray = nu.zeros((timeSpreadPast, resolution), dtype=nu.int)
                #         for i, _t in enumerate(targetIndices[:-1]):
                #             lowerBound = data.loc[_t, 'Low']
                #             upperBound = data.loc[_t, 'High']
                #             graphArray[i, :] = nu.array([ True if lowerBound <= value <= upperBound else False for value in topDownArray ]) * 1
                #         graphData = pd.DataFrame(graphArray, index=data.loc[targetIndices[:-1], 'Date'])
                #         graphName = data.loc[t, 'Date'].split('.')[0].replace('T', '_').replace(':', '-')
                #         graphData.to_csv(f'{graphDataDir}/{graphName}.csv', index=True, header=True)
                data.to_csv(storedFilePath, index=False, header=True)


        elif span == '1HOUR':
            if startDate != None and os.path.exists(storedFilePath):
                fileNames = filter(lambda name: '.csv' in name and os.stat(f'{dirName}/{name}').st_size > 10, os.listdir(dirName))
                fileNames = [ (name, dt.datetime.strptime(name.split('.csv')[0], '%Y_%m_%d_%H_%M')) for name in fileNames ]
                fileNames = filter(lambda pair: pair[1] >= startDate, fileNames)
                fileNames = sorted(fileNames, key=lambda pair: pair[1])
                # fetch old data
                data = pd.read_csv(storedFilePath)
                data['DateTypeDate'] = stringToDate(data['Date'].values.ravel())
                # fetch new data
                ts = nu.array([], dtype=nu.int)
                for name, date in fileNames:
                    new15minData = pd.read_csv(f'{dirName}/{name}')
                    # drop last row
                    new15minData = newData.drop(newData.index[[-1]])
                    new15minData['DateTypeDate'] = stringToDate(new15minData['Date'].values.ravel())
                    # dump 15minData into 1HourData
                    hourRoundedDate = dt.datetime(date.year, date.month, date.day, date.hour, 0, 0)
                    # init newHourData
                    newHourData = new15minData.copy()
                    newHourData.drop(newHourData.index[:])
                    for row in new15minData.index:
                        # if the row is within 1hour since last, dump it into the row
                        if new15minData.loc[row, 'DateTypeDate'].hour == hourRoundedDate.hour:
                            newHourData.loc[row, 'High'] = max(newHourData.loc[row, 'High'], new15minData.loc[row, 'High'])
                            newHourData.loc[row, 'Low'] = min(newHourData.loc[row, 'Low'], new15minData.loc[row, 'Low'])
                            # only for the last one. For convenience, we propose it for all
                            newHourData.loc[row, 'Close'] = new15minData.loc[row, 'Close']
                            newHourData.loc[row, 'time_close'] = new15minData.loc[row, 'time_close']
                        # should create new row for the next hour
                        else:
                            hourRoundedDate += dt.timedelta(hours=1)
                            newHourData.loc[row] = new15minData.loc[row]
                            newHourData.loc['time_period_end'] = dateToString(hourRoundedDate + dt.timedelta(hours=1))
                    # get the specific row of the old data to insert into
                    rowIndex = data.loc[data['Date'] == dateToString(date)].index.values
                    if rowIndex.shape[0] != 0:
                        newIndices = range(rowIndex[0], rowIndex[0] + newHourData.index.values.shape[0])
                        for i, row in enumerate(newIndices):
                            data.loc[row] = newHourData.loc[i]
                        ts = nu.concatenate([ts, newIndices])
                    else: # add to the last line
                        newIndices = range(data.index.values.shape[0], data.index.values.shape[0] + newHourData.index.values.shape[0])
                        for i, row in enumerate(newIndices):
                            data.loc[row] = newHourData.loc[i]
                        ts = nu.concatenate([ts, newIndices])
                        data = data.sort_values('DateTypeDate').reset_index(drop=True)
                # calculate graphData and label
                data = generateGraphDataAndLabel(data=data, ts=ts, resolution=resolution, timeSpreadPast=int(timeSpreadPast), timeSpreadFuture=int(timeSpreadFuture), graphDataDir=graphDataDir)
                del data['DateTypeDate']
                data.to_csv(storedFilePath, index=False, header=True)

            else: # don't have start Date
                fileNames = filter(lambda name: '.csv' in name and os.stat(f'{dirName}/{name}').st_size > 10, os.listdir(dirName))
                fileNames = [ (name, dt.datetime.strptime(name.split('.csv')[0], '%Y_%m_%d_%H_%M')) for name in fileNames ]
                fileNames = sorted(fileNames, key=lambda pair: pair[1])
                # # fetch the first data
                # name, date = fileNames[0]
                # data = pd.read_csv(f'{dirName}/{name}')
                # for name, date in fileNames[1:]:
                #     _newData = pd.read_csv(f'{dirName}/{name}')
                #     _newData = _newData.drop(_newData.index[[-1]])
                #     data = pd.concat([data, _newData])
                data = None
                # fetch new data
                for name, date in fileNames:
                    new15minData = pd.read_csv(f'{dirName}/{name}')
                    # drop last row
                    new15minData = newData.drop(newData.index[[-1]])
                    new15minData['DateTypeDate'] = stringToDate(new15minData['Date'].values.ravel())
                    # dump 15minData into 1HourData
                    hourRoundedDate = dt.datetime(date.year, date.month, date.day, date.hour, 0, 0)
                    # init newHourData
                    newHourData = new15minData.copy()
                    newHourData.drop(newHourData.index[:])
                    for row in new15minData.index:
                        # if the row is within 1hour since last, dump it into the row
                        if new15minData.loc[row, 'DateTypeDate'].hour == hourRoundedDate.hour:
                            newHourData.loc[row, 'High'] = max(newHourData.loc[row, 'High'], new15minData.loc[row, 'High'])
                            newHourData.loc[row, 'Low'] = min(newHourData.loc[row, 'Low'], new15minData.loc[row, 'Low'])
                            # only for the last one. For convenience, we propose it for all
                            newHourData.loc[row, 'Close'] = new15minData.loc[row, 'Close']
                            newHourData.loc[row, 'time_close'] = new15minData.loc[row, 'time_close']
                        # should create new row for the next hour
                        else:
                            hourRoundedDate += dt.timedelta(hours=1)
                            newHourData.loc[row] = new15minData.loc[row]
                            newHourData.loc['time_period_end'] = dateToString(hourRoundedDate + dt.timedelta(hours=1))
                    # if old data is empty
                    if data is None:
                        data = newHourData.copy()
                    else:
                        data = pd.concat([data, newHourData])
                data = data.drop(['Volume', 'trades_count'], axis=1).dropna().reset_index(drop=True)
                data['LabelCNNPost1'] = nu.nan
                del fileNames
                # calculate graphData and update label
                data = generateGraphDataAndLabel(data=data, ts=data.index.values.ravel(), resolution=resolution, timeSpreadPast=int(timeSpreadPast), timeSpreadFuture=int(timeSpreadFuture), graphDataDir=graphDataDir)
                data.to_csv(storedFilePath, index=False, header=True)

        timeCost = (dt.datetime.now() - _start).total_seconds()
        print(f'History data preprocessing done with time cost: {timeCost} seconds')
        return data


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


    def downloadAndUpdateHistoryDataToLatest(self, shouldCalculateLabelsFromBegining):
        dirName = './HistoryData'
        fileNames = filter(lambda name: '.csv' in name and os.stat(f'{dirName}/{name}').st_size > 10, os.listdir(dirName))
        fileNames = sorted([ (name, dt.datetime.strptime(name.split('.csv')[0], '%Y_%m_%d_%H_%M')) for name in fileNames ], key=lambda pair: pair[1])
        storedDirName = './LabeledData'
        storedFilePath = f'{storedDirName}/15MIN.csv'
        graphDataDir = f'./{storedDirName}/graphData'
        # dowload data
        start = fileNames[-2][1]
        now = dt.datetime.utcnow()
        self.download15MinuteSpanData(start=start, end=now)
        # calculate 15MIN.csv
        if shouldCalculateLabelsFromBegining:
            data = self.processShortermHistoryData(startDate=None, resolution=self.resolution, timeSpreadPast=self.timeSpreadPast)
            return data
        else:
            data = self.processShortermHistoryData(startDate=start, resolution=self.resolution, timeSpreadPast=self.timeSpreadPast)
            return data


def generateGraphDataAndLabel(data, ts, resolution, timeSpreadPast, timeSpreadFuture, graphDataDir):
    for t in ts:
        # if data down to -timeSpreadPast is not available, skip drawing graph.
        if not t+1-timeSpreadPast in data.index.values:
            continue
        # if data up to +timeSpreadFuture is available, label it.
        if t+timeSpreadFuture in data.index.values:
            # prepare for labeling
            nowMiddle = (data.loc[t, 'High'] + data.loc[t, 'Low'])/2
            futureMiddle = (data.loc[t+1:t+1+timeSpreadFuture, 'High'].values.ravel().mean() + data.loc[t+1:t+1+timeSpreadFuture, 'Low'].values.ravel().mean())/2
            sigma = nu.abs((data.loc[t, 'High'] - data.loc[t, 'Low'])/(1.96*2))
            if futureMiddle > nowMiddle + 1.96*sigma:
                data.loc[t, 'LabelCNNPost1'] = 0
            elif futureMiddle >= nowMiddle - 1.96*sigma:
                data.loc[t, 'LabelCNNPost1'] = 1
            else:
                data.loc[t, 'LabelCNNPost1'] = 2
        # draw graph:
        targetIndices = range(t+1-timeSpreadPast, t+1)
        top = data.loc[targetIndices, 'High'].max()
        down = data.loc[targetIndices, 'Low'].min()
        topDownArray = nu.linspace(down, top, resolution)
        graphArray = nu.zeros((timeSpreadPast, resolution), dtype=nu.int)
        for i, _t in enumerate(targetIndices):
            lowerBound = data.loc[_t, 'Low']
            upperBound = data.loc[_t, 'High']
            graphArray[i, :] = nu.array([ True if lowerBound <= value <= upperBound else False for value in topDownArray ]) * 1
        graphData = pd.DataFrame(graphArray, index=data.loc[targetIndices, 'Date'])
        graphName = data.loc[t, 'Date'].split('.')[0].replace('T', '_').replace(':', '-')
        graphData.to_csv(f'{graphDataDir}/{graphName}.csv', index=True, header=True)
    return data


def dateToString(date):
    return date.strftime('%Y-%m-%d') + 'T' + date.strftime('%H:%M:%S') + '.0000000Z'


def stringToDate(dateString):
    if type(dateString) is nu.ndarray:
        return nu.array([dt.datetime.strptime(d.split('.')[0].replace('T', '_').replace(':', '-'), "%Y-%m-%d_%H-%M-%S") for d in dateString])

    else:
        return dt.datetime.strptime(dateString.split('.')[0].replace('T', '_').replace(':', '-'), "%Y-%m-%d_%H-%M-%S")



if __name__ == '__main__':
    worker = PreprocessingWorker(resolution=int(24*8), timeSpreadPast=int(24*8))
    # worker.process()
    # worker.showData(path='./LabeledData/15MIN.csv')

    # start = dt.datetime(2020,  7, 15, 0, 0, 0)
    # now = dt.datetime.utcnow()
    # worker.download15MinuteSpanData(start=start, end=now)

    # worker.dumpShortermDataIntoSpanData()

    # worker.processShortermHistoryData(span='15MIN')

    worker.downloadAndUpdateHistoryDataToLatest(shouldCalculateLabelsFromBegining=False)
