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
from CNNAI import CNNAI


# Model

def getGraphData(name, span):
    return (name, pd.read_csv(f'./LabeledData/{span}/graphData/{name}', index_col=0).values)


class BTC_JPY_Environment(py_environment.PyEnvironment):
    def __init__(self, imageWidth, imageHeight, initialAsset, dtype=nu.float32, isHugeMemorryMode=True, shouldGiveRewardsFinally=True, gamma=0.99):
        self.dtype = dtype
        self.__actionSpec = BoundedArraySpec(shape=(1,), dtype=self.dtype, minimum=0, maximum=1, name='action')
        # https://www.tensorflow.org/agents/api_docs/python/tf_agents/environments/py_environment/PyEnvironment#observation_spec
        # https://www.tensorflow.org/agents/api_docs/python/tf_agents/typing/types/NestedArraySpec
        # self.__observationSpec = (BoundedArraySpec(shape=(imageWidth, imageHeight), dtype=self.dtype, minimum=0, maximum=1, name='observation_market'), BoundedArraySpec(shape=(1,), dtype=self.dtype, minimum=0, maximum=1, name='observation_holdingRate'))
        # self.__observationSpec = {
        #     'observation_predictedCategory': BoundedArraySpec(shape=(1,), dtype=self.dtype, minimum=-1, maximum=1, name='observation_predictedCategory'),
        #     'observation_holdingRate': BoundedArraySpec(shape=(1,), dtype=self.dtype, minimum=0, maximum=1, name='observation_holdingRate')
        # }
        # self.savedElements = int(3*4*48) # 3elements * 4quaterPerHour * 48Hour
        # self.__observationSpec = BoundedArraySpec(shape=(self.savedElements+1,), dtype=self.dtype, minimum=0, maximum=1, name='observation')
        self.__observationSpec = BoundedArraySpec(shape=(7,), dtype=self.dtype, minimum=0, maximum=1, name='observation')
        self.holdingBTC = 0.0
        self.holdingJPY = float(initialAsset)
        self.initialAsset = float(initialAsset)
        self.holdingRate = 0.0
        # self.currentState = (nu.zeros((imageWidth, imageHeight), dtype=self.dtype), nu.array([self.holdingRate], dtype=self.dtype))

        self.currentState = nu.zeros(7, dtype=self.dtype)
        self.currentOpenPrice = 0.0
        self.currentClosePrice = 0.0

        self.imageWidth = imageWidth
        self.imageHeight = imageHeight

        self.episodeEndSteps = 1*7*24*4  # 1week * 7days/week * 24hours/day * 4quater/hour

        self.data = {}
        self.data['15MIN'] = pd.read_csv(f'./LabeledData/15MIN/labeledData.csv').dropna(axis=1)
        self.data['15MIN']['DateTypeDate'] = stringToDate(self.data['15MIN']['Date'].values.ravel())
        self.possibleStartDate = [ pd.Timestamp(nu_date).to_pydatetime() for nu_date in self.data['15MIN'].iloc[:-self.episodeEndSteps*2, :].loc[:, 'DateTypeDate'].values.ravel() ]
        startDate = nu.random.choice(self.possibleStartDate)
        self.currentDate = dt.datetime(startDate.year, startDate.month, startDate.day, startDate.hour, (startDate.minute//15)*15, 0)
        self.data['1HOUR'] = pd.read_csv(f'./LabeledData/1HOUR/labeledData.csv').dropna(axis=1)
        self.data['1HOUR']['DateTypeDate'] = stringToDate(self.data['1HOUR']['Date'].values.ravel())
        self.data['1HOUR_interpolated'] = pd.read_csv(f'./LabeledData/1HOUR/labeledData_interpolated.csv').dropna(axis=1)
        self.data['1HOUR_interpolated']['DateTypeDate'] = stringToDate(self.data['1HOUR_interpolated']['Date'].values.ravel())

        self.episodeCount = 0#
        # reward will be clipped to [-1, 1] using reward/(coeff*initAsset)
        self.rewardClipCoeff = 1.0

        self.isHugeMemorryMode = isHugeMemorryMode
        if isHugeMemorryMode:
            self.graphData = {}
            with mp.Pool(processes=min(mp.cpu_count()-1, 8)) as pool:
                filesAndSpans = []
                for span in ['15MIN', '1HOUR']:
                    files = list(filter(lambda path: path.split('.')[1] == 'csv', os.listdir(f'./LabeledData/{span}/graphData')))
                    filesAndSpans.extend([ (file, span) for file in files ])
                    self.graphData[f'{span}'] = pool.starmap(getGraphData, filesAndSpans)
                    self.graphData[f'{span}'] = { name: graphData for name, graphData in self.graphData[f'{span}'] }
            # self.graphData = { path: pd.read_csv(f'./LabeledData/graphData/{path}', index_col=0).values for path in os.listdir('./LabeledData/graphData') if path.split('.')[1] == 'csv' }
        else:
            self.graphData = None

        self.shouldGiveRewardsFinally = shouldGiveRewardsFinally
        self.gamma = gamma

        # set cnnAIs
        self.cnnAIs = {}
        for span in ['15MIN', '1HOUR']:
            # set model of 15MIN / 1HOUR
            modelPath = f'cnnmodel{span}.h5'
            if os.path.exists(modelPath):
                model = tf.keras.models.load_model(modelPath)
                self.cnnAIs[f'{span}'] = CNNAI(span=span, resolution=imageWidth, timeSpreadPast=imageHeight, model=model)
            else:
                print(f"{span} model not given, start training ...")
                self.cnnAIs[f'{span}'] = CNNAI(span=span, resolution=imageWidth, timeSpreadPast=imageHeight, model=None)
                self.cnnAIs[f'{span}'].train()


    # required
    def action_spec(self):
        return self.__actionSpec


    # required
    def observation_spec(self):
        return self.__observationSpec


    # required
    def _reset(self):
        self.holdingBTC = 0.0
        self.holdingJPY = self.initialAsset

        # get available current date
        _didFindNextTime = False
        while not _didFindNextTime:
            _didFindNextTime = True
            self.currentDate = nu.random.choice(self.possibleStartDate)
            for span in ['15MIN', '1HOUR']:
                # check if 15MIN/1HOUR span graph data exists
                _graphPath = f'./LabeledData/{span}/graphData/' + self.currentDate.strftime('%Y-%m-%d_%H-%M-%S') + '.csv'
                if not os.path.exists(_graphPath):
                    _didFindNextTime = False
        # set current price and holding rate
        self.currentOpenPrice = self.data['15MIN'].loc[self.data['15MIN']['DateTypeDate']==self.currentDate, 'Open'].values[0]
        self.currentClosePrice = self.data['15MIN'].loc[self.data['15MIN']['DateTypeDate']==self.currentDate, 'Close'].values[0]
        self.holdingRate = 0.0
        # loop in spans
        self.currentState = nu.array([], dtype=self.dtype)
        # self.currentState = nu.zeros(self.savedElements, dtype=self.dtype)
        for span in ['15MIN', '1HOUR']:
            # get next market snapshot
            if self.isHugeMemorryMode:
                _graphPath = self.currentDate.strftime('%Y-%m-%d_%H-%M-%S') + '.csv'
                marketSnapshot = self.graphData[f'{span}'][f'{_graphPath}']
            else:
                _graphPath = f'./LabeledData/{span}/graphData/' + self.currentDate.strftime('%Y-%m-%d_%H-%M-%S') + '.csv'
                marketSnapshot = pd.read_csv(_graphPath, index_col=0).values
            marketSnapshot = marketSnapshot.astype(self.dtype)
            predictedProbabilities = self.cnnAIs[span].predictFromCurrentGraphData(data=marketSnapshot, now=self.currentDate, shouldSaveGraph=False)
            for _probability in predictedProbabilities.ravel():
                self.currentState = nu.append(self.currentState, _probability)

            # self.currentState = nu.concatenate([self.currentState, predictedProbabilities.ravel()])
            # self.currentState[-3:] = predictedProbabilities.ravel()
        # self.currentState = nu.append(marketSnapshot, self.holdingRate)
        # self.currentState = (marketSnapshot, nu.array([self.holdingRate], dtype=self.dtype))
        # self.currentState = {
        #     'observation_predictedCategory': predictedCategory,
        #     'observation_holdingRate': nu.array([self.holdingRate], dtype=self.dtype)
        # }
        # self.currentState = nu.array([predictedProbabilities[0], predictedProbabilities[1], predictedProbabilities[2], self.holdingRate], dtype=self.dtype)
        self.currentState = nu.append(self.currentState, self.holdingRate).astype(self.dtype)
        self.episodeCount = 0
        return ts.restart(self.currentState)


    # required
    def _step(self, action):
        if self.__checkIfEpisodeShouldEnd()  == True:
            if self.shouldGiveRewardsFinally:
                reward = (self.currentClosePrice * self.holdingBTC + self.holdingJPY - self.initialAsset) / (self.rewardClipCoeff*self.initialAsset)
                # print('Episode did ended at step {} with reward: {}'.format(self.episodeCount, reward))
                return ts.termination(self.currentState, reward)
            else:
                return ts.termination(self.currentState, 0)
        # if should continue trading
        self.episodeCount += 1
        # get next time
        _didFindNextTime = False
        while not _didFindNextTime:
            self.currentDate += dt.timedelta(minutes=15)
            _didFindNextTime = True
            for span in ['15MIN', '1HOUR']:
                interpolated_span = f'{span}_interpolated' if span != '15MIN' else span
                # get next data
                currentData = self.data[interpolated_span].loc[self.data[interpolated_span]['DateTypeDate']==self.currentDate, :]
                if len(currentData['Open'].values.ravel()) == 0:
                    _didFindNextTime = False
                    continue
                # check if 15MIN span graph data exists
                _graphPath = f'./LabeledData/{span}/graphData/' + self.currentDate.strftime('%Y-%m-%d_%H-%M-%S') + '.csv'
                if not os.path.exists(_graphPath):
                    _didFindNextTime = False
        self.currentOpenPrice = currentData['Open'].values.ravel()[0]
        self.currentClosePrice = currentData['Close'].values.ravel()[0]
        newState = nu.array([], dtype=self.dtype)
        # newState = nu.roll(self.currentState, -3)
        # get next market snapshot
        for span in ['15MIN', '1HOUR']:
        # for span in ['15MIN']:
            _graphDir = f'./LabeledData/{span}/graphData'
            if self.isHugeMemorryMode:
                _graphPath = self.currentDate.strftime('%Y-%m-%d_%H-%M-%S') + '.csv'
                nextMarketSnapshot = self.graphData[f'{span}'][f'{_graphPath}']
            else:
                _graphPath = f'{_graphDir}/' + self.currentDate.strftime('%Y-%m-%d_%H-%M-%S') + '.csv'
                nextMarketSnapshot = pd.read_csv(_graphPath, index_col=0).values
            nextMarketSnapshot = nextMarketSnapshot.astype(self.dtype)
            predictedProbabilities = self.cnnAIs[f'{span}'].predictFromCurrentGraphData(data=nextMarketSnapshot, now=self.currentDate, shouldSaveGraph=False)
            newState = nu.concatenate([newState, predictedProbabilities.ravel()])
            # newState[-4:-1] = predictedProbabilities.ravel()
        # # get next holding rate according to specific action taken
        # # action[0] = percentage of selling(+) holdingJPY / selling(-) holdingBTC
        # # action[1] = exchanging rate (relatively to current rate)
        # price = self.currentOpenPrice * (1+action[1])
        # exchangeIndicator = action[0]
        # # store asset before action
        # assetBeforeAction = self.currentOpenPrice * self.holdingBTC + self.holdingJPY
        # # if should add some BTC
        # if exchangeIndicator > 0:
        #     # if have any JPY left
        #     if self.holdingRate < 1:
        #         costJPY = self.holdingJPY * exchangeIndicator
        #         # if btc can be bought, update holdings; otherwise do nothing
        #         if currentData['Low'].values[0] < price:
        #             self.holdingBTC += costJPY / price
        #             self.holdingJPY -= costJPY
        #             self.holdingRate = self.holdingBTC*self.currentClosePrice / (self.holdingJPY + self.holdingBTC*self.currentClosePrice)
        # # if should sell some BTC
        # elif exchangeIndicator < 0:
        #     # if have any BTC to be sold
        #     if self.holdingRate > 0:
        #         costBTCAmount = self.holdingBTC * (-exchangeIndicator)
        #         # if btc can be sold, update holdings; otherwise do nothing
        #         if price < currentData['High'].values[0]:
        #             self.holdingBTC -= costBTCAmount
        #             self.holdingJPY += costBTCAmount * price
        #             self.holdingRate = self.holdingBTC*self.currentClosePrice / (self.holdingJPY + self.holdingBTC*self.currentClosePrice)
        # else:
        #     pass  # do nothing if deltaHoldingRate == 0

        # translate actions to specific Coincheck API command
        # action[0] = target holding rate
        targetHoldingRate = action[0]
        # store asset before action
        assetBeforeAction = self.currentOpenPrice * self.holdingBTC + self.holdingJPY
        # if should add some BTC
        if targetHoldingRate > self.holdingRate:
            # if have any JPY left
            if self.holdingRate < 1:
                addBTCAmount = (targetHoldingRate - self.holdingRate) * (self.holdingBTC + self.holdingJPY/self.currentOpenPrice)
                # addBTCAmout = max(addBTCAmount, 0) # should above 0
                addBTCAmount = min(self.holdingJPY/self.currentOpenPrice, addBTCAmount)
                assert addBTCAmount >= 0, f'targetHoldingRate = {targetHoldingRate}, self.holdingRate = {self.holdingRate}, addBTCAmount = {addBTCAmount}, self.holdingJPY = {self.holdingJPY}, self.currentOpenPrice = {self.currentOpenPrice}'
                # if btc can be bought, update holdings; otherwise do nothing
                if currentData['Low'].values[0] <= self.currentOpenPrice:
                    costJPY = addBTCAmount * self.currentOpenPrice
                    self.holdingBTC += costJPY / self.currentOpenPrice
                    self.holdingJPY -= costJPY
                    self.holdingRate = self.holdingBTC*self.currentClosePrice / (self.holdingJPY + self.holdingBTC*self.currentClosePrice)
        # if should sell some BTC
        elif targetHoldingRate < self.holdingRate:
            # if have any BTC to be sold
            if self.holdingRate > 0:
                sellBTCAmount = (self.holdingRate - targetHoldingRate) * (self.holdingBTC + self.holdingJPY/self.currentOpenPrice)
                # sellBTCAmout = max(sellBTCAmount, 0) # should above 0
                sellBTCAmount = min(self.holdingBTC, sellBTCAmount)
                assert sellBTCAmount >= 0, f'targetHoldingRate = {targetHoldingRate}, self.holdingRate = {self.holdingRate}, sellBTCAmount = {sellBTCAmount}, self.holdingBTC = {self.holdingBTC}'
                # if btc can be sold, update holdings; otherwise do nothing
                if self.currentOpenPrice < currentData['High'].values[0]:
                    self.holdingBTC -= sellBTCAmount
                    self.holdingJPY += sellBTCAmount * self.currentOpenPrice
                    self.holdingRate = self.holdingBTC*self.currentClosePrice / (self.holdingJPY + self.holdingBTC*self.currentClosePrice)
        else:
            pass  # do nothing if targetHoldingRate == self.holdingRate

        # get next holding rate according to specific action taken


        # concate marketData and holdingRate to make currentState
        # self.currentState = (nextMarketSnapshot, nu.array([self.holdingRate], dtype=self.dtype))
        # self.currentState = {
        #     'observation_predictedCategory': predictedCategory,
        #     'observation_holdingRate': nu.array([self.holdingRate], dtype=self.dtype)
        # }
        # self.currentState = nu.array([predictedProbabilities[0], predictedProbabilities[1], predictedProbabilities[2], self.holdingRate], dtype=self.dtype)

        self.currentState = nu.append(newState, self.holdingRate).astype(self.dtype)
        # newState[-1] = self.holdingRate
        # self.currentState = newState.copy().astype(self.dtype)
        # returns
        if not self.shouldGiveRewardsFinally:
            assetAfterAction = self.currentClosePrice * self.holdingBTC + self.holdingJPY
            deltaAsset = assetAfterAction - assetBeforeAction
            _stepReward = deltaAsset/(self.rewardClipCoeff*self.initialAsset)
            # print('steps: {:>4}; state: down(15min): {:+4.2f}, level(15min): {:+4.2f}, up(15min): {:+4.2f}, down(1hour): {:+4.2f}, level(1hour): {:+4.2f}, up(1hour): {:+4.2f}, holdingRate: {:.4f}; buy/sell amount of BTC: {:+5.2f}@rate: {:+5.2f}; BTC: {:.3f}, JPY: {:>8}, asset: {:>8}, reward: {:+6.3f}'.format(self.episodeCount, self.currentState[0], self.currentState[1], self.currentState[2], self.currentState[3], self.currentState[4], self.currentState[5], self.currentState[6], action[0], action[1], self.holdingBTC, self.holdingJPY, self.currentOpenPrice*self.holdingBTC+self.holdingJPY, _stepReward))
            return ts.transition(self.currentState, reward=_stepReward, discount=self.gamma)
        else:
            # print('steps: {:>4}; state: down(15min): {:+4.2f}, level(15min): {:+4.2f}, up(15min): {:+4.2f}, down(1hour): {:+4.2f}, level(1hour): {:+4.2f}, up(1hour): {:+4.2f}, holdingRate: {:.4f}; buy/sell amount of BTC: {:+5.2f}@rate: {:+5.2f}; BTC: {:.3f}, JPY: {:>8}, asset: {:>8}'.format(self.episodeCount, self.currentState[0], self.currentState[1], self.currentState[2], self.currentState[3], self.currentState[4], self.currentState[5], self.currentState[6], action[0], action[1], self.holdingBTC, self.holdingJPY, self.currentOpenPrice*self.holdingBTC+self.holdingJPY))
            return ts.transition(self.currentState, reward=0, discount=self.gamma)


    def __checkIfEpisodeShouldEnd(self):
        didBankrupted = (self.currentOpenPrice * self.holdingBTC + self.holdingJPY) <= self.initialAsset * 0.5
        # print(f'didBankrupted = {didBankrupted}')
        return didBankrupted or self.episodeCount > self.episodeEndSteps



# Main

if __name__ == '__main__':
    startDate = dt.datetime(2018,  7, 15, 0, 0, 0)
    env = BTC_JPY_Environment(imageWidth=int(24*8), imageHeight=int(24*8), initialAsset=100000, isHugeMemorryMode=False)

    utils.validate_py_environment(env, episodes=3)

    # constantAction = nu.array([0.5, 0.0])
    # timeStep = env.reset()
    # print(timeStep)
    # cumulatedReward = timeStep.reward
    # for _ in range(24*4):
    #     timeStep = env.step(constantAction)
    #     print(timeStep)
    #     cumulatedReward += timeStep.reward
