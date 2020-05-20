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


bitflyerBaseURL = 'https://api.bitflyer.com/v1/'
shouldContinue = True


class TradingAgent():
    def __init__(self):
        self.client = bf.API()
        self.processes = []

    def run(self):
        responseQueue = mp.SimpleQueue()

        process = mp.Process(target=listenMarketWithMinTimeSpan, args=(self.client, responseQueue))
        process.start()
        self.processes.append(process)

        process = mp.Process(target=translateAccumulatedResponses, args=(responseQueue,))
        process.start()
        self.processes.append(process)


        try:
            while True:
                time.sleep(10)

        except KeyboardInterrupt:
            print('Start ending process ...')
            shouldContinue = False
            for process in self.processes:
                process.terminate()

        except concurrent.futures.InvalidStateError:
            print('Error: some tasks have finished unexpectedly.')
            shouldContinue = False
            for process in self.processes:
                process.terminate()

        finally:
            print('Shutting down ... (This may take several minutes)')


def listenMarketWithMinTimeSpan(client, queue):
    print('Start listening at bitflyer market in 5 second span ...')
    while True:
        if shouldContinue == False:
            return

        now = dt.datetime.utcnow()
        if 0 <= float(now.second) % 5 < 1:
            response = client.ticker(product_code="BTC_JPY")
            queue.put(response)
            time.sleep(1)


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


if __name__ == '__main__':
    agent = TradingAgent()
    agent.run()
