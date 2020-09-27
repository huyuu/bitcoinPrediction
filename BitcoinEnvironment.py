# Import Modules
# Python modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import tensorflow as tf
import numpy as nu
import pandas as pd
import datetime as dt
import os
import multiprocessing as mp
# tensorflows
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.specs.array_spec import BoundedArraySpec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
tf.compat.v1.enable_v2_behavior()
# custom modules
from PreprocessingWorker import stringToDate, dateToString


# Model

def getGraphData(path):
    return (path, pd.read_csv(f'./LabeledData/graphData/{path}', index_col=0).values)


class BTC_JPY_Environment(py_environment.PyEnvironment):
    def __init__(self, imageWidth, imageHeight, initialAsset, dtype=nu.float32, isHugeMemorryMode=True):
        self.dtype = dtype
        self.__actionSpec = BoundedArraySpec(shape=(2,), dtype=self.dtype, minimum=-1, maximum=1, name='action')
        # https://www.tensorflow.org/agents/api_docs/python/tf_agents/environments/py_environment/PyEnvironment#observation_spec
        # https://www.tensorflow.org/agents/api_docs/python/tf_agents/typing/types/NestedArraySpec
        # self.__observationSpec = (BoundedArraySpec(shape=(imageWidth, imageHeight), dtype=self.dtype, minimum=0, maximum=1, name='observation_market'), BoundedArraySpec(shape=(1,), dtype=self.dtype, minimum=0, maximum=1, name='observation_holdingRate'))
        self.__observationSpec = {
            'observation_market': BoundedArraySpec(shape=(imageWidth, imageHeight), dtype=self.dtype, minimum=0, maximum=1, name='observation_market'),
            'observation_holdingRate': BoundedArraySpec(shape=(1,), dtype=self.dtype, minimum=0, maximum=1, name='observation_holdingRate')
        }
        self.holdingBTC = 0
        self.holdingJPY = initialAsset
        self.initialAsset = initialAsset
        self.holdingRate = 0.0
        self.currentState = (nu.zeros((imageWidth, imageHeight), dtype=self.dtype), nu.array([self.holdingRate], dtype=self.dtype))
        self.currentPrice = 0.0

        self.imageWidth = imageWidth
        self.imageHeight = imageHeight

        self.data = pd.read_csv('./LabeledData/15MIN.csv').dropna(axis=1)
        self.data['DateTypeDate'] = stringToDate(self.data['Date'].values.ravel())
        self.possibleStartDate = [ pd.Timestamp(nu_date).to_pydatetime() for nu_date in self.data.iloc[:-int(4*24*31*3), :].loc[:, 'DateTypeDate'].values.ravel() ]
        startDate = nu.random.choice(self.possibleStartDate)
        self.currentDate = dt.datetime(startDate.year, startDate.month, startDate.day, startDate.hour, (startDate.minute//15)*15, 0)

        self.episodeCount = 0
        self.episodeEndSteps = 4*24*30*3

        self.isHugeMemorryMode = isHugeMemorryMode
        if isHugeMemorryMode:
            with mp.Pool(processes=min(mp.cpu_count()-1, 8)) as pool:
                files = list(filter(lambda path: path.split('.')[1] == 'csv', os.listdir('./LabeledData/graphData')))
                self.graphData = pool.map(getGraphData, files)
                self.graphData = { data[0]: data[1] for data in self.graphData }
            # self.graphData = { path: pd.read_csv(f'./LabeledData/graphData/{path}', index_col=0).values for path in os.listdir('./LabeledData/graphData') if path.split('.')[1] == 'csv' }
        else:
            self.graphData = None


    # required
    def action_spec(self):
        return self.__actionSpec


    # required
    def observation_spec(self):
        return self.__observationSpec


    # required
    def _reset(self):
        self.holdingBTC = 0
        self.holdingJPY = self.initialAsset
        self.holdingRate = 0.0
        self.currentDate = nu.random.choice(self.possibleStartDate)
        self.currentPrice = 0.0
        # get next market snapshot
        _graphDir = './LabeledData/graphData'
        if self.isHugeMemorryMode:
            _graphPath = self.currentDate.strftime('%Y-%m-%d_%H-%M-%S') + '.csv'
            marketSnapshot = self.graphData[f'{_graphPath}']
        else:
            _graphPath = f'{_graphDir}/' + self.currentDate.strftime('%Y-%m-%d_%H-%M-%S') + '.csv'
            marketSnapshot = pd.read_csv(_graphPath, index_col=0).values
        marketSnapshot = marketSnapshot.astype(self.dtype)
        # self.currentState = nu.append(marketSnapshot, self.holdingRate)
        # self.currentState = (marketSnapshot, nu.array([self.holdingRate], dtype=self.dtype))
        self.currentState = {
            'observation_market': marketSnapshot,
            'observation_holdingRate': nu.array([self.holdingRate], dtype=self.dtype)
        }
        self.episodeCount = 0
        return ts.restart(self.currentState)


    # required
    def _step(self, action):
        if self.__checkIfEpisodeShouldEnd()  == True:
            reward = self.currentPrice * self.holdingBTC + self.holdingJPY - self.initialAsset
            print('Episode did ended with reward: {}'.format(reward))
            return ts.termination(self.currentState, reward)
        # if should continue trading
        self.episodeCount += 1
        while True:
            # get next time
            self.currentDate += dt.timedelta(minutes=15)
            # get next data
            nextData = self.data.loc[self.data['DateTypeDate']==self.currentDate, :]
            if len(nextData['Open'].values.ravel()) != 0:
                break
            else:
                continue
        self.currentPrice = nextData['Open'].values.ravel()[0]
        nextClosePrice = nextData['Close'].values.ravel()[0]
        # get next market snapshot
        _graphDir = './LabeledData/graphData'
        if self.isHugeMemorryMode:
            _graphPath = self.currentDate.strftime('%Y-%m-%d_%H-%M-%S') + '.csv'
            nextMarketSnapshot = self.graphData[f'{_graphPath}']
        else:
            _graphPath = f'{_graphDir}/' + self.currentDate.strftime('%Y-%m-%d_%H-%M-%S') + '.csv'
            nextMarketSnapshot = pd.read_csv(_graphPath, index_col=0).values
        nextMarketSnapshot = nextMarketSnapshot.astype(self.dtype)
        # get next holding rate according to specific action taken
        price = self.currentPrice * (1+action[1])
        exchangeIndicator = action[0]
        # if should add some BTC
        if exchangeIndicator > 0:
            if self.holdingRate < 1:
                costJPY = self.holdingJPY * exchangeIndicator
                # if btc can be bought, update holdings; otherwise do nothing
                if nextData['Low'].values[0] < price:
                    self.holdingBTC += costJPY / price
                    self.holdingJPY -= costJPY
                    self.holdingRate = self.holdingBTC*nextClosePrice / (self.holdingJPY + self.holdingBTC*nextClosePrice)
        # if should sell some BTC
        elif exchangeIndicator < 0:
            costBTCAmount = self.holdingBTC * (-exchangeIndicator)
            # if btc can be sold, update holdings; otherwise do nothing
            if price < nextData['High'].values[0]:
                self.holdingBTC -= costBTCAmount
                self.holdingJPY += costBTCAmount * price
                self.holdingRate = self.holdingBTC*nextClosePrice / (self.holdingJPY + self.holdingBTC*nextClosePrice)
        else:
            pass  # do nothing if deltaHoldingRate == 0

        if 0 <= self.holdingRate <= 1:
            print(self.holdingRate)
            raise ValueError
        # concate marketData and holdingRate to make currentState
        # self.currentState = (nextMarketSnapshot, nu.array([self.holdingRate], dtype=self.dtype))
        self.currentState = {
            'observation_market': nextMarketSnapshot,
            'observation_holdingRate': nu.array([self.holdingRate], dtype=self.dtype)
        }
        print('holdingRate: {:.3g}, holdingBTC: {:.4g}, holdingJPY: {:.4g}'.format(self.holdingRate, self.holdingBTC, self.holdingJPY))
        return ts.transition(self.currentState, reward=0, discount=1.0)


    def __checkIfEpisodeShouldEnd(self):
        didBankrupted = self.holdingBTC <= 1e-4 and self.holdingJPY <= 100.0
        return didBankrupted or self.episodeCount > self.episodeEndSteps



# Main

if __name__ == '__main__':
    startDate = dt.datetime(2018,  7, 15, 0, 0, 0)
    env = BTC_JPY_Environment(imageWidth=int(24*4), imageHeight=int(24*8), initialAsset=100000, isHugeMemorryMode=False)

    utils.validate_py_environment(env, episodes=3)

    # constantAction = nu.array([0.5, 0.0])
    # timeStep = env.reset()
    # print(timeStep)
    # cumulatedReward = timeStep.reward
    # for _ in range(24*4):
    #     timeStep = env.step(constantAction)
    #     print(timeStep)
    #     cumulatedReward += timeStep.reward
