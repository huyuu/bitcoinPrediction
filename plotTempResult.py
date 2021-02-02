import numpy as nu
import pickle
from matplotlib import pyplot as pl


with open("SACAgent_tempResults.pickle", "rb") as file:
    result = pickle.load(file)

pl.plot(result[:, 0], result[:, 1])
pl.show()

