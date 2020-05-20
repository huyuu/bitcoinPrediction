import pybitflyer as bf
import multiprocessing as mp
import time
import datetime as dt
import concurrent.futures
import pandas as pd
import numpy as nu
import queue as qu
import os


bitflyerBaseURL = 'https://api.bitflyer.com/v1/'
shouldContinue = True


class TradingAgent():
    def __init__(self):
        self.client = bf.API()
        # self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=3)
        self.processes = []

    def run(self):
        responseQueue = mp.SimpleQueue()
        # listenTask = self.executor.submit(listenMarketWithMinTimeSpan)
        # translationTask = self.executor.submit(translateAccumulatedResponses)
        process = mp.Process(target=listenMarketWithMinTimeSpan, args=(self.client, responseQueue))
        process.start()
        self.processes.append(process)
        process = mp.Process(target=translateAccumulatedResponses, args=(responseQueue,))
        process.start()
        self.processes.append(process)
        try:
            while True:
                # if listenTask.done() or translationTask.done():
                #     raise concurrent.futures.InvalidStateError
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
            # self.executor.shutdown(wait=True)



def listenMarketWithMinTimeSpan(client, queue):
    global responseQueue
    print('Start listening at bitflyer market in 5 second span ...')
    while True:
        if shouldContinue == False:
            return

        now = dt.datetime.utcnow()
        if 0 <= float(now.second) % 5 <= 1:
            response = client.ticker(product_code="BTC_JPY")
            queue.put(response)
            time.sleep(2)


def translateAccumulatedResponses(queue):
    global responseQueue
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
            data.to_csv(filePath)
            print(f'Translation ends after processing {responseCount} responses.')
            # sleep for 1 minute
            time.sleep(60 * 4)


if __name__ == '__main__':
    agent = TradingAgent()
    agent.run()
