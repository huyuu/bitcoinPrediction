from matplotlib import pyplot as pl
from matplotlib import cm
from mpl_toolkits import mplot3d
import mplfinance as mpf
import matplotlib.dates as mdates
import pandas as pd


def plot():
    path = "BTC-JPY.csv"
    data = pd.read_csv(path).head(150).dropna()

    data.index = pd.to_datetime(data['Date'])


    # fig = pl.figure()
    # ax = pl.axes()
    labelPlot = mpf.make_addplot(data['ClassLabel'], scatter=True)
    rsi14Plot = mpf.make_addplot(data['RSI14'], panel='lower')
    BBup = mpf.make_addplot(data['BB+1sigma'])
    BBmiddle = mpf.make_addplot(data['BBmiddle'])
    BBlow = mpf.make_addplot(data['BB-1sigma'])
    mpf.plot(data, type='candle', addplot=[BBup, BBmiddle, BBlow, labelPlot])
    pl.show()


if __name__ == '__main__':
    plot()
