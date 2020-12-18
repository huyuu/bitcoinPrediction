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
        # get file names from ./HistoryData
        fileNames = filter(lambda name: '.csv' in name and os.stat(f'{dirName}/{name}').st_size > 10, os.listdir(dirName))
        fileNames = [ (name, dt.datetime.strptime(name.split('.csv')[0], '%Y_%m_%d_%H_%M')) for name in fileNames ]
        fileNames = sorted(fileNames, key=lambda pair: pair[1])


        # 15MIN Data
        storedFilePath = f'{storedDirName}/15MIN/labeledData.csv'
        graphDataDir = f'./{storedDirName}/15MIN/graphData'
        if not os.path.exists(graphDataDir):
            os.mkdir(graphDataDir)
        if startDate != None and os.path.exists(storedFilePath):
            filteredFileNames = filter(lambda pair: pair[1] >= startDate, fileNames)
            # fetch old data
            data = pd.read_csv(storedFilePath)
            data['DateTypeDate'] = stringToDate(data['Date'].values.ravel())
            # fetch new data
            ts = nu.array([], dtype=nu.int)
            for name, date in filteredFileNames:
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
            # data = generateGraphDataAndLabel(data=data, ts=ts, resolution=resolution, timeSpreadPast=int(timeSpreadPast), timeSpreadFuture=int(timeSpreadFuture), graphDataDir=graphDataDir)
            # del data['DateTypeDate']
            data.to_csv(storedFilePath, index=False, header=True)

        else: # don't have startDate
            # fetch data
            name, date = fileNames[0]
            data = pd.read_csv(f'{dirName}/{name}')
            data['DateTypeDate'] = stringToDate(data['Date'].values.ravel())
            for name, date in fileNames[1:]:
                _newData = pd.read_csv(f'{dirName}/{name}')
                _newData = _newData.drop(_newData.index[[-1]])
                _newData['DateTypeDate'] = stringToDate(_newData['Date'].values.ravel())
                data = pd.concat([data, _newData])
            data = data.drop(['Volume', 'trades_count'], axis=1).dropna().reset_index(drop=True)
            data['LabelCNNPost1'] = nu.nan
            # del data['DateTypeDate']
            # calculate graphData and update label
            # data = generateGraphDataAndLabel(data=data, ts=data.index.values.ravel(), resolution=resolution, timeSpreadPast=int(timeSpreadPast), timeSpreadFuture=int(timeSpreadFuture), graphDataDir=graphDataDir)
            data.to_csv(storedFilePath, index=False, header=True)


        # 1HOUR Data (normal)
        storedFilePath = f'{storedDirName}/1HOUR/labeledData.csv'
        graphDataDir = f'./{storedDirName}/1HOUR/graphData'
        if not os.path.exists(graphDataDir):
            os.mkdir(graphDataDir)
        if startDate != None and os.path.exists(storedFilePath):
            # fetch old data
            data1HOUR = pd.read_csv(storedFilePath)
            data1HOUR['DateTypeDate'] = stringToDate(data1HOUR['Date'].values.ravel())
            # fetch new data
            ts = nu.array([], dtype=nu.int)
            for name, date in filteredFileNames:
                new15minData = pd.read_csv(f'{dirName}/{name}')
                # drop last row
                new15minData = new15minData.drop(new15minData.index[[-1]])
                new15minData['DateTypeDate'] = stringToDate(new15minData['Date'].values.ravel())
                # dump 15minData into 1HourData
                hourRoundedDate = dt.datetime(date.year, date.month, date.day, date.hour, 0, 0)
                # init newHourData
                newHourData = new15minData.copy()
                # drop all the rows except the first
                newHourData = newHourData.drop(newHourData.index[1:])
                for row in new15minData.index:
                    # if the row is within 1hour since last, dump it into the row
                    if new15minData.loc[row, 'DateTypeDate'].hour == hourRoundedDate.hour:
                        lastIndex = newHourData.index[-1]
                        newHourData.loc[lastIndex, 'High'] = max(newHourData.loc[lastIndex, 'High'], new15minData.loc[row, 'High'])
                        newHourData.loc[lastIndex, 'Low'] = min(newHourData.loc[lastIndex, 'Low'], new15minData.loc[row, 'Low'])
                        # only for the last one. For convenience, we propose it for all
                        newHourData.loc[lastIndex, 'Close'] = new15minData.loc[row, 'Close']
                        newHourData.loc[lastIndex, 'time_close'] = new15minData.loc[row, 'time_close']
                    # should create new row for the next hour
                    else:
                        hourRoundedDate += dt.timedelta(hours=1)
                        newHourData = newHourData.append(new15minData.loc[row], ignore_index=True)
                        lastIndex = newHourData.index[-1]
                        newHourData.loc[lastIndex, 'time_period_end'] = dateToString(hourRoundedDate + dt.timedelta(hours=1))
                # get the specific row of the old data to insert into
                rowIndex = data1HOUR.loc[data1HOUR['Date'] == dateToString(date)].index.values
                if rowIndex.shape[0] != 0:
                    newIndices = range(rowIndex[0], rowIndex[0] + newHourData.index.values.shape[0])
                    for i, row in enumerate(newIndices):
                        data1HOUR.loc[row] = newHourData.loc[i]
                    ts = nu.concatenate([ts, newIndices])
                else: # add to the last line
                    newIndices = range(data1HOUR.index.values.shape[0], data1HOUR.index.values.shape[0] + newHourData.index.values.shape[0])
                    for i, row in enumerate(newIndices):
                        data1HOUR.loc[row] = newHourData.loc[i]
                    ts = nu.concatenate([ts, newIndices])
                    data1HOUR = data1HOUR.sort_values('DateTypeDate').reset_index(drop=True)
            # calculate graphData and label
            # data = generateGraphDataAndLabel(data=data, ts=ts, resolution=resolution, timeSpreadPast=int(timeSpreadPast), timeSpreadFuture=int(timeSpreadFuture), graphDataDir=graphDataDir)
            # del data1HOUR['DateTypeDate']
            data1HOUR.to_csv(storedFilePath, index=False, header=True)

        else: # don't have start Date
            data1HOUR = None
            # fetch new data
            for name, date in fileNames:
                new15minData = pd.read_csv(f'{dirName}/{name}')
                # drop last row
                new15minData = new15minData.drop(new15minData.index[[-1]])
                new15minData['DateTypeDate'] = stringToDate(new15minData['Date'].values.ravel())
                # dump 15minData into 1HourData
                hourRoundedDate = dt.datetime(date.year, date.month, date.day, date.hour, 0, 0)
                # hourRoundedDate = dt.datetime(new15minData.loc[0, 'DateTypeDate'].year, new15minData.loc[0, 'DateTypeDate'].month, new15minData.loc[0, 'DateTypeDate'].day, new15minData.loc[0, 'DateTypeDate'].hour, 0, 0)
                # init newHourData
                newHourData = new15minData.copy()
                # drop all the rows except the first
                newHourData = newHourData.drop(newHourData.index[1:])
                # update the first row's time_period_end to the start of the next hour
                newHourData.loc[0, 'time_period_end'] = dateToString(hourRoundedDate + dt.timedelta(hours=1))
                for row in new15minData.index[1:]:
                    # if the row is within 1hour since last, dump it into the row
                    if new15minData.loc[row, 'DateTypeDate'].hour == hourRoundedDate.hour:
                        lastIndex = newHourData.index[-1]
                        newHourData.loc[lastIndex, 'High'] = max(newHourData.loc[lastIndex, 'High'], new15minData.loc[row, 'High'])
                        newHourData.loc[lastIndex, 'Low'] = min(newHourData.loc[lastIndex, 'Low'], new15minData.loc[row, 'Low'])
                        # only for the last one. For convenience, we propose it for all
                        newHourData.loc[lastIndex, 'Close'] = new15minData.loc[row, 'Close']
                        newHourData.loc[lastIndex, 'time_close'] = new15minData.loc[row, 'time_close']
                    # should create new row for the next hour
                    else:
                        hourRoundedDate += dt.timedelta(hours=1)
                        newHourData = newHourData.append(new15minData.loc[row], ignore_index=True)
                        lastIndex = newHourData.index[-1]
                        newHourData.loc[lastIndex, 'time_period_end'] = dateToString(hourRoundedDate + dt.timedelta(hours=1))
                # for the 1st time we copy newHourData, otherwise we concatenate them
                data1HOUR = newHourData.copy() if data1HOUR is None else pd.concat([data1HOUR, newHourData])
            data1HOUR = data1HOUR.drop(['Volume', 'trades_count'], axis=1).dropna().reset_index(drop=True)
            data1HOUR['LabelCNNPost1'] = nu.nan
            # del fileNames
            # calculate graphData and update label
            # del data1HOUR['DateTypeDate']
            data1HOUR.to_csv(storedFilePath, index=False, header=True)
            # generateGraphDataAndLabel(data=data, ts=data.index.values.ravel(), resolution=resolution, timeSpreadPast=int(timeSpreadPast), timeSpreadFuture=int(timeSpreadFuture), graphDataDir=graphDataDir)


        # 1HOUR_interpolated
        storedFilePath = f'{storedDirName}/1HOUR/labeledData_interpolated.csv'
        if startDate != None and os.path.exists(storedFilePath):
            # fetch old data
            data1HOUR_interpolated = pd.read_csv(storedFilePath)
            data1HOUR_interpolated['DateTypeDate'] = stringToDate(data1HOUR_interpolated['Date'].values.ravel())
            # fetch new data
            for name, date in filteredFileNames:
                new15minData = pd.read_csv(f'{dirName}/{name}')
                # drop last row
                new15minData = new15minData.drop(new15minData.index[[-1]])
                new15minData['DateTypeDate'] = stringToDate(new15minData['Date'].values.ravel())
                # dump 15minData into 1HourData
                hourRoundedDate = dt.datetime(date.year, date.month, date.day, date.hour, 0, 0)
                # init newHourData
                newHourData = new15minData.copy()
                # drop all the rows except the first
                newHourData = newHourData.drop(newHourData.index[1:])
                for row in new15minData.index:
                    # if the row is not hourRounded
                    if new15minData.loc[row, 'DateTypeDate'].hour == hourRoundedDate.hour:
                        newHourData = newHourData.append(new15minData.loc[row], ignore_index=True)
                        lastIndex = newHourData.index[-1]
                        previousIndex = newHourData.indx[-2]
                        #
                        newHourData.loc[lastIndex, 'Open'] = newHourData.loc[previousIndex, 'Open']
                        newHourData.loc[lastIndex, 'time_open'] = newHourData.loc[previousIndex, 'time_open']
                        newHourData.loc[lastIndex, 'time_period_start'] = dateToString(hourRoundedDate)
                        newHourData.loc[lastIndex, 'time_period_end'] = dateToString(hourRoundedDate + dt.timedelta(hours=1))
                        # adjust High & Low
                        newHourData.loc[lastIndex, 'High'] = max(newHourData.loc[previousIndex, 'High'], newHourData.loc[lastIndex, 'High'])
                        newHourData.loc[lastIndex, 'Low'] = min(newHourData.loc[previousIndex, 'Low'], newHourData.loc[lastIndex, 'Low'])
                    # should create new row for the next hour
                    else:
                        hourRoundedDate += dt.timedelta(hours=1)
                        newHourData = newHourData.append(new15minData.loc[row], ignore_index=True)
                        lastIndex = newHourData.index[-1]
                        newHourData.loc[lastIndex, 'time_period_end'] = dateToString(hourRoundedDate + dt.timedelta(hours=1))
                        newHourData.loc[lastIndex, 'time_period_start'] = dateToString(hourRoundedDate)
                # get the specific row of the old data to insert into
                rowIndex = data1HOUR_interpolated.loc[data1HOUR_interpolated['Date'] == dateToString(date)].index.values
                if rowIndex.shape[0] != 0: # if rowIndex exists
                    newIndices = range(rowIndex[0], rowIndex[0] + newHourData.index.values.shape[0])
                    for i, row in enumerate(newIndices):
                        data1HOUR_interpolated.loc[row] = newHourData.loc[i]
                    # ts = nu.concatenate([ts, newIndices])
                else: # add to the last line
                    newIndices = range(data1HOUR.index.values.shape[0], data1HOUR.index.values.shape[0] + newHourData.index.values.shape[0])
                    for i, row in enumerate(newIndices):
                        data1HOUR_interpolated.loc[row] = newHourData.loc[i]
                    # ts = nu.concatenate([ts, newIndices])
                    data1HOUR_interpolated = data1HOUR_interpolated.sort_values('DateTypeDate').reset_index(drop=True)
            # del data1HOUR_interpolated['DateTypeDate']
            data1HOUR_interpolated.to_csv(storedFilePath, index=False, header=True)

        else: # don't have start Date
            data1HOUR_interpolated = None
            # fetch new data
            for name, date in fileNames:
                new15minData = pd.read_csv(f'{dirName}/{name}')
                # drop last row
                new15minData = new15minData.drop(new15minData.index[[-1]])
                new15minData['DateTypeDate'] = stringToDate(new15minData['Date'].values.ravel())
                # dump 15minData into 1HourData
                hourRoundedDate = dt.datetime(date.year, date.month, date.day, date.hour, 0, 0)
                # hourRoundedDate = dt.datetime(new15minData.loc[0, 'DateTypeDate'].year, new15minData.loc[0, 'DateTypeDate'].month, new15minData.loc[0, 'DateTypeDate'].day, new15minData.loc[0, 'DateTypeDate'].hour, 0, 0)
                # init newHourData
                newHourData = new15minData.copy()
                # drop all the rows except the first
                newHourData = newHourData.drop(newHourData.index[1:])
                # update the first row's time_period_end to the start of the next hour
                newHourData.loc[0, 'time_period_start'] = dateToString(hourRoundedDate)
                newHourData.loc[0, 'time_period_end'] = dateToString(hourRoundedDate + dt.timedelta(hours=1))
                for row in new15minData.index[1:]:
                    # if the row is not hourRounded
                    if new15minData.loc[row, 'DateTypeDate'].hour == hourRoundedDate.hour:
                        newHourData = newHourData.append(new15minData.loc[row], ignore_index=True)
                        lastIndex = newHourData.index[-1]
                        previousIndex = newHourData.index[-2]
                        #
                        newHourData.loc[lastIndex, 'Open'] = newHourData.loc[previousIndex, 'Open']
                        newHourData.loc[lastIndex, 'time_open'] = newHourData.loc[previousIndex, 'time_open']
                        newHourData.loc[lastIndex, 'time_period_start'] = dateToString(hourRoundedDate)
                        newHourData.loc[lastIndex, 'time_period_end'] = dateToString(hourRoundedDate + dt.timedelta(hours=1))
                        # adjust High & Low
                        newHourData.loc[lastIndex, 'High'] = max(newHourData.loc[previousIndex, 'High'], newHourData.loc[lastIndex, 'High'])
                        newHourData.loc[lastIndex, 'Low'] = min(newHourData.loc[previousIndex, 'Low'], newHourData.loc[lastIndex, 'Low'])
                    # should create new row for the next hour
                    else:
                        hourRoundedDate += dt.timedelta(hours=1)
                        newHourData = newHourData.append(new15minData.loc[row], ignore_index=True)
                        lastIndex = newHourData.index[-1]
                        newHourData.loc[lastIndex, 'time_period_end'] = dateToString(hourRoundedDate + dt.timedelta(hours=1))
                        newHourData.loc[lastIndex, 'time_period_start'] = dateToString(hourRoundedDate)
                # for the 1st time we copy newHourData, otherwise we concatenate them
                data1HOUR_interpolated = newHourData.copy() if data1HOUR_interpolated is None else pd.concat([data1HOUR_interpolated, newHourData])
            data1HOUR_interpolated = data1HOUR_interpolated.drop(['Volume', 'trades_count'], axis=1).dropna().reset_index(drop=True)
            data1HOUR_interpolated['LabelCNNPost1'] = nu.nan
            # store
            # del data1HOUR_interpolated['DateTypeDate']
            data1HOUR_interpolated.to_csv(storedFilePath, index=False, header=True)
        # calculate graphData and update label
        generateGraphDataAndLabel(data, data1HOUR, data1HOUR_interpolated, resolution, timeSpreadPast, timeSpreadFuture=timeSpreadPast//2)
        # calcuate time
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


    def downloadAndUpdateHistoryDataToLatest(self, shouldCalculateLabelsFromBegining, shouldConductDownload=True):
        dirName = './HistoryData'
        fileNames = filter(lambda name: '.csv' in name and os.stat(f'{dirName}/{name}').st_size > 10, os.listdir(dirName))
        fileNames = sorted([ (name, dt.datetime.strptime(name.split('.csv')[0], '%Y_%m_%d_%H_%M')) for name in fileNames ], key=lambda pair: pair[1])
        # storedDirName = './LabeledData'
        # storedFilePath = f'{storedDirName}/{span}.csv'
        # graphDataDir = f'./{storedDirName}/graphData'
        # dowload data
        start = fileNames[-2][1]
        now = dt.datetime.utcnow()
        if shouldConductDownload:
            self.download15MinuteSpanData(start=start, end=now)
        # calculate 15MIN.csv
        if shouldCalculateLabelsFromBegining:
            self.processShortermHistoryData(startDate=None, resolution=self.resolution, timeSpreadPast=self.timeSpreadPast)
        else:
            self.processShortermHistoryData(startDate=start, resolution=self.resolution, timeSpreadPast=self.timeSpreadPast)



def generateGraphDataAndLabel(data15MIN, data1HOUR, data1HOUR_interpolated, resolution, timeSpreadPast, timeSpreadFuture, pointsPerCandle=10, determinantPriceDiversityPercentage=0.05):
    # for t in data15MIN.index[timeSpreadPast+1: -timeSpreadFuture]:
    for t in data15MIN.index[1:]:
        currentClosePrice = data15MIN.loc[t, 'Close']
        currentTime = data15MIN.loc[t, 'DateTypeDate']
        # decide price of growing and dropping
        determinantPriceOfGrowing = currentClosePrice * (1 + determinantPriceDiversityPercentage)
        determinantPriceOfDropping = currentClosePrice * (1 - determinantPriceDiversityPercentage)
        # label 15MINData
        # get future average price from now to now+timeSpreadFuture
        if t+timeSpreadFuture in data15MIN.index.values:
            futureAverage = 0
            for futureT in range(t+1, t+1+timeSpreadFuture):
                _futurePricesInFutureT = nu.linspace(data15MIN.loc[futureT, 'Low'], data15MIN.loc[futureT, 'High'], pointsPerCandle)
                for _futurePrice in _futurePricesInFutureT:
                    futureAverage += _futurePrice
            futureAverage /= float(pointsPerCandle*timeSpreadFuture)
            # labeling
            if futureAverage >= determinantPriceOfGrowing: # growing
                data15MIN.loc[t, 'LabelCNNPost1'] = 2
            elif futureAverage <= determinantPriceOfDropping: # dropping
                data15MIN.loc[t, 'LabelCNNPost1'] = 0
            else: # level
                data15MIN.loc[t, 'LabelCNNPost1'] = 1
        # draw past graph from now-timeSpreadPast+1 to now
        if t+1-timeSpreadPast in data15MIN.index.values:
            targetIndices = range(t+1-timeSpreadPast, t+1)
            top = data15MIN.loc[targetIndices, 'High'].max()
            down = data15MIN.loc[targetIndices, 'Low'].min()
            topDownArray = nu.linspace(down, top, resolution)
            graphArray = nu.zeros((timeSpreadPast, resolution), dtype=nu.int)
            for i, _t in enumerate(targetIndices):
                lowerBound = data15MIN.loc[_t, 'Low']
                upperBound = data15MIN.loc[_t, 'High']
                graphArray[i, :] = nu.array([ True if lowerBound <= value <= upperBound else False for value in topDownArray ]) * 1
            graphData = pd.DataFrame(graphArray.T, index=topDownArray, columns=data15MIN.loc[targetIndices, 'Date'])
            graphName = data15MIN.loc[t, 'Date'].split('.')[0].replace('T', '_').replace(':', '-')
            graphData.to_csv(f'./LabeledData/15MIN/graphData/{graphName}.csv', index=True, header=True)

        # label 1HOURData
        # get future average price from now to now+timeSpreadFuture_1hour
        currentHourRoundedTime = dt.datetime(currentTime.year, currentTime.month, currentTime.day, currentTime.hour, 0, 0)
        try:
            t_1hour = data1HOUR.loc[data1HOUR['DateTypeDate'] == currentHourRoundedTime].index[0]
        except IndexError as e:
            print(f'{currentHourRoundedTime} is not in data1HOUR')
            exit()
        try:
            t_1hour_interpolated = data1HOUR_interpolated.loc[data1HOUR_interpolated['DateTypeDate'] == currentTime].index[0]
        except:
            print(f'{currentTime} is not in data1HOUR_interpolated')
            exit()
        # t_1hour_interpolated = data1HOUR_interpolated.index[data1HOUR_interpolated['DateTypeDate'] == currentTime]
        if t_1hour+(timeSpreadFuture+1) in data1HOUR.index.values:
            futureAverage = 0
            for futureT in range(t_1hour+1, t_1hour+1+timeSpreadFuture):
                _futurePricesInFutureT = nu.linspace(data1HOUR.loc[futureT, 'Low'], data1HOUR.loc[futureT, 'High'], pointsPerCandle)
                for _futurePrice in _futurePricesInFutureT:
                    futureAverage += _futurePrice
            futureAverage /= float(pointsPerCandle*timeSpreadFuture)
            # labeling
            if futureAverage >= determinantPriceOfGrowing: # growing
                data1HOUR.loc[t_1hour, 'LabelCNNPost1'] = 2
            elif futureAverage <= determinantPriceOfDropping: # dropping
                data1HOUR.loc[t_1hour, 'LabelCNNPost1'] = 0
            else: # level
                data1HOUR.loc[t_1hour, 'LabelCNNPost1'] = 1
        # draw past graph from now-timeSpreadPast+1 to now
        if t_1hour+1-timeSpreadPast in data1HOUR.index.values:
            targetIndicesInFull1HOURData = range(t_1hour+1-timeSpreadPast, t_1hour)
            top = data1HOUR.loc[targetIndicesInFull1HOURData, 'High'].max()
            top = max(top, data1HOUR_interpolated.loc[t_1hour_interpolated, 'High'])
            down = data1HOUR.loc[targetIndicesInFull1HOURData, 'Low'].min()
            down = min(down, data1HOUR_interpolated.loc[t_1hour_interpolated, 'Low'])
            topDownArray = nu.linspace(down, top, resolution)
            graphArray = nu.zeros((timeSpreadPast, resolution), dtype=nu.int)
            # create full 1 hour history data graph (t_1hour+1-timeSpreadPast ~ t_1hour) from data1HOUR
            for i, _t in enumerate(targetIndicesInFull1HOURData):
                _lowerBound = data1HOUR.loc[_t, 'Low']
                _upperBound = data1HOUR.loc[_t, 'High']
                graphArray[i, :] = nu.array([ True if _lowerBound <= value <= _upperBound else False for value in topDownArray ]) * 1
            # create 1 hour data interpolated
            _lowerBound = data1HOUR_interpolated.loc[t_1hour_interpolated, 'Low']
            _upperBound = data1HOUR_interpolated.loc[t_1hour_interpolated, 'High']
            graphArray[-1, :] = nu.array([ True if _lowerBound <= value <= _upperBound else False for value in topDownArray ]) * 1
            # save
            graphData = pd.DataFrame(graphArray.T, index=topDownArray, columns=data1HOUR.loc[range(t_1hour+1-timeSpreadPast, t_1hour+1), 'Date'])
            graphName = data1HOUR.loc[t, 'Date'].split('.')[0].replace('T', '_').replace(':', '-')
            graphData.to_csv(f'./LabeledData/1HOUR/graphData/{graphName}.csv', index=True, header=True)
    # for t in ts:
    #     # if data down to -timeSpreadPast is not available, skip drawing graph.
    #     if not t+1-timeSpreadPast in data.index.values:
    #         continue
    #     # if data up to +timeSpreadFuture is available, label it.
    #     if t+timeSpreadFuture in data.index.values:
    #         # prepare for labeling
    #         # nowMiddle = (data.loc[t, 'High'] + data.loc[t, 'Low'])/2
    #         nowClose = data.loc[t, 'Close']
    #         futureMiddle = (data.loc[t+1:t+1+timeSpreadFuture, 'High'].values.ravel().mean() + data.loc[t+1:t+1+timeSpreadFuture, 'Low'].values.ravel().mean())/2
    #         sigma = nu.abs((data.loc[t, 'High'] - data.loc[t, 'Low'])/(1.96*2))
    #         if futureMiddle > nowClose + 1.96*sigma:
    #             data.loc[t, 'LabelCNNPost1'] = 0
    #         elif futureMiddle >= nowClose - 1.96*sigma:
    #             data.loc[t, 'LabelCNNPost1'] = 1
    #         else:
    #             data.loc[t, 'LabelCNNPost1'] = 2
    #     # draw graph:
    #     targetIndices = range(t+1-timeSpreadPast, t+1)
    #     top = data.loc[targetIndices, 'High'].max()
    #     down = data.loc[targetIndices, 'Low'].min()
    #     topDownArray = nu.linspace(down, top, resolution)
    #     graphArray = nu.zeros((timeSpreadPast, resolution), dtype=nu.int)
    #     for i, _t in enumerate(targetIndices):
    #         lowerBound = data.loc[_t, 'Low']
    #         upperBound = data.loc[_t, 'High']
    #         graphArray[i, :] = nu.array([ True if lowerBound <= value <= upperBound else False for value in topDownArray ]) * 1
    #     graphData = pd.DataFrame(graphArray, index=data.loc[targetIndices, 'Date'])
    #     graphName = data.loc[t, 'Date'].split('.')[0].replace('T', '_').replace(':', '-')
    #     graphData.to_csv(f'{graphDataDir}/{graphName}.csv', index=True, header=True)
    #

    del data15MIN['DateTypeDate']
    data15MIN.to_csv('./LabeledData/15MIN/labeledData.csv', index=False, header=True)
    del data1HOUR['DateTypeDate']
    data1HOUR.to_csv('./LabeledData/1HOUR/labeledData.csv', index=False, header=True)
    del data1HOUR_interpolated['DateTypeDate']
    data1HOUR_interpolated.to_csv('./LabeledData/1HOUR/labeledData_interpolated.csv', index=False, header=True)
    return


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

    worker.downloadAndUpdateHistoryDataToLatest(shouldCalculateLabelsFromBegining=True, shouldConductDownload=False)
