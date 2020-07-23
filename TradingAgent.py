import multiprocessing as mp
import subprocess as sp
import pip
import time
import datetime as dt
import concurrent.futures
import pandas as pd
import numpy as nu
import queue as qu
import os
import pybitflyer as bf

# from NeuralNetworkAI import NeuralNetworkAI
from CNNAI import CNNAI
from PreprocessingWorker import PreprocessingWorker, dateToString, stringToDate


bitflyerBaseURL = 'https://api.bitflyer.com/v1/'
shouldContinue = True


class TradingAgent():
    def __init__(self):
        self.processes = []


    def run(self):
        responseQueue = mp.SimpleQueue()
        # run listen market process
        process = mp.Process(target=listenMarketWithMinTimeSpan, args=(responseQueue,))
        process.start()
        self.processes.append(process)
        # # run translate accumulated responses process
        # process = mp.Process(target=translateAccumulatedResponses, args=(responseQueue,))
        # process.start()
        # self.processes.append(process)
        # # run predict now process
        # process = mp.Process(target=predictTheLatest15MIN)
        # process.start()
        # self.processes.append(process)

        # main loop
        try:
            while True:
                time.sleep(3600 * 24)

        except KeyboardInterrupt:
            print('Start ending process ...')
            shouldContinue = False
            time.sleep(8)
            for process in self.processes:
                process.terminate()

        except concurrent.futures.InvalidStateError:
            print('Error: some tasks have finished unexpectedly.')
            shouldContinue = False
            for process in self.processes:
                process.terminate()

        finally:
            print('Shutting down ... (This may take several minutes)')


def listenMarketWithMinTimeSpan(queue):
    # preprocessing
    client = bf.API()
    ai = CNNAI()
    worker = PreprocessingWorker(resolution=ai.resolution, timeSpreadPast=ai.timeSpreadPast)
    data = worker.downloadAndUpdateHistoryDataToLatest(shouldCalculateLabelsFromBegining=False)
    data['DateTypeDate'] = stringToDate(data['Date'].values.ravel())
    # start listening market
    print('Start listening at bitflyer market in 5 second span ...')
    while True:
        if shouldContinue == False:
            data = data.sort_values('DateTypeDate').reset_index(drop=True)
            del data['DateTypeDate']
            data.to_csv('./LabeledData/15MIN.csv', index=False, header=True)
            print('15MIN.csv updated.')
            return
        # update current time
        now = dt.datetime.utcnow()
        if 0 <= float(now.second) % 5 < 1:
            # get response
            response = client.ticker(product_code="BTC_JPY")
            # if access failed, continue
            if not 'timestamp' in response:
                time.sleep(2)
                continue
            # get response time region
            responseTime = dt.datetime.strptime(response['timestamp'].split('.')[0].replace('T', '_').replace(':', '-'), '%Y-%m-%d_%H-%M-%S')
            responseTimeString = dateToString(responseTime)
            _minute = (responseTime.minute // 15) * 15
            responseTimeRegion = dateToString(dt.datetime(responseTime.year, responseTime.month, responseTime.day, responseTime.hour, _minute, 0))
            # get current value
            currentValue = response['ltp']
            # search for response time region in data
            rowIndex = data.loc[data['Date'] == responseTimeRegion].index
            # if row exists, update; otherwise create one
            if rowIndex.values.shape[0] != 0:
                data.loc[rowIndex, 'time_close'] = responseTimeString
                data.loc[rowIndex, 'High'] = max(data.loc[rowIndex, 'High'].values[0], currentValue)
                data.loc[rowIndex, 'Low'] = min(data.loc[rowIndex, 'Low'].values[0], currentValue)
                data.loc[rowIndex, 'Close'] = currentValue
            else:
                # translate response to newData
                newData = {
                    'Date': responseTimeRegion,
                    'time_period_end': dateToString(dt.datetime(responseTime.year, responseTime.month, responseTime.day, responseTime.hour, _minute, 0) + dt.timedelta(minutes=15)),
                    'time_open': responseTimeString,
                    'time_close': responseTimeString,
                    'Open': currentValue,
                    'High': currentValue,
                    'Low': currentValue,
                    'Close': currentValue,
                    'DateTypeDate': dt.datetime(responseTime.year, responseTime.month, responseTime.day, responseTime.hour, _minute, 0)
                }
                data = data.append(newData, ignore_index=True)
            # predict
            ai.predictFromCurrentData(data, now, graphDataDir='./StoredData')
            # save every time
            data = data.sort_values('DateTypeDate').reset_index(drop=True)
            data.to_csv('./LabeledData/15MIN.csv', columns=['Date', 'time_period_end', 'time_open', 'time_close', 'Open', 'High', 'Low', 'Close', 'LabelCNNPost1'], index=False, header=True)
            # queue.put(response)
            time.sleep(1)
        else:
            time.sleep(0.5)


def translateAccumulatedResponses(queue):
    if not os.path.exists("./StoredData"):
        os.mkdir("StoredData")

    while True:
        if shouldContinue == False:
            return

        now = dt.datetime.utcnow()
        if 0 <= int(now.minute) % 5 <= 1:
            print('Start translating responses ...')
            filePath = './StoredData/' + now.strftime('%Y_%m_%d') + '.csv'
            # get initial data
            responseCount = 0
            if os.path.exists(filePath):
                data = pd.read_csv(filePath)
            else:
                if queue.empty():
                    print('Translation ends with empty.')
                    time.sleep(60 * 4)
                    continue
                responseCount += 1
                data = pd.DataFrame([queue.get()])
            # process current responses
            while not queue.empty():
                responseCount += 1
                response = queue.get()
                data = pd.concat([data, pd.DataFrame([response])])
            data.to_csv(filePath, index=False)
            print(f'Translation ends after processing {responseCount} responses.')
            # sleep for 1 minute
            time.sleep(60 * 4)
        else:
            time.sleep(10)


def predictTheLatest15MIN():
    ai = NeuralNetworkAI()
    preprocessingWorker = PreprocessingWorker()

    while True:
        if shouldContinue == False:
            return

        now = dt.datetime.utcnow()
        # now = dt.datetime(2020, 5, 24, 3, 16, 0)
        if 1 <= float(now.minute) % 15 < 2:
            # get rawData from path
            dirName = './StoredData'
            nowString = now.strftime('%Y_%m_%d')
            path = f'{dirName}/{nowString}.csv'
            preprocessingWorker.dumpShortermDataIntoSpanData(span='15MIN', end=now)
            # rawData = pd.read_csv(path).dropna()
            # set Date
            # rawData['Date'] = nu.array([dt.datetime.fromisoformat(str.split('.')[0]) for str in rawData['timestamp']])
            # _end = dt.datetime(now.year, now.month, now.day, now.hour, (int(now.minute)//15)*15, now.second)
            # _start = _end - dt.timedelta(minutes=15)
            # _targets = rawData[rawData.Date >= _start][rawData.Date <= _end]['ltp'].values
            # _open = _targets[0]
            # _close = _targets[-1]
            # _high = _targets.max()
            # _low = _targets.min()
            # data = pd.DataFrame({
            #     'Open': _open,
            #     'Close': _close,
            #     'High': _high,
            #     'Low': _low
            # }, index=[0])
            # del rawData, _start, _end, _targets, _open, _close, _high, _low
            # data['RSI14'] = ta.RSI(data['Close'], timeperiod=14)
            # data['BB+2sigma'], data['BBmiddle'], data['BB-2sigma'] = ta.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
            # data['Sigma'] = (data['BB+2sigma'] - data['BBmiddle'])/2
            # data['BBPosition'] = (data['Close'] - data['BBmiddle'])/data['Sigma']
            data = pd.read_csv('./LabeledData/15MIN.csv').tail(1)
            # pass it to ai
            ai.predictCurrentSituation(data, shouldShowGraph=False)
            time.sleep(60 * 13)
        else:
            time.sleep(10)



if __name__ == '__main__':
    agent = TradingAgent()
    agent.run()
    # predictTheLatest15MIN()
