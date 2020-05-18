import numpy as nu
import pandas as pd
import sklearn as sk
from sklearn import gaussian_process as gp
from matplotlib import pyplot as pl
from matplotlib import cm
from mpl_toolkits import mplot3d


def correctFunction(a, b):
    return nu.sqrt(a**2 + b**2)

correctFunction_v = nu.vectorize(correctFunction)

upperBound = 100.0
trainPoints = 30
plotPoints = 50


trainSamples = nu.random.rand(trainPoints, 2) * upperBound
trainLabels = correctFunction_v(trainSamples[:, 0], trainSamples[:, 1]).ravel()
_testSamples = nu.linspace(0, upperBound, plotPoints)
_x, _y = nu.meshgrid(_testSamples, _testSamples, indexing='ij')
testSamples = nu.concatenate([_x.reshape(-1, 1), _y.reshape(-1, 1)], axis=1)

constantKernel = gp.kernels.ConstantKernel(1.0)
rbfKernel = gp.kernels.RBF(100.0)
kernel = constantKernel * rbfKernel
model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)

model.fit(trainSamples, trainLabels)
print(f'Model: {model.kernel_}')
print(f'Log_Marginal_Likelihood: {model.log_marginal_likelihood(model.kernel_.theta)}')

testLabels, std = model.predict(testSamples, return_std=True)
testLabels = testLabels.reshape(plotPoints, plotPoints)
std = std.reshape(plotPoints, plotPoints)

print("std:")
print(std)

xs = _testSamples[:, nu.newaxis]
ys = _testSamples[nu.newaxis, :]
zs = testLabels + std
fig = pl.figure()
ax = pl.axes(projection='3d')
ax.plot_surface(xs, ys, zs, alpha=0.8, cmap='gray')

xs = trainSamples[:, 0]
ys = trainSamples[:, 1]
zs = trainLabels
ax.scatter3D(xs, ys, zs, linewidths=3)
pl.show()
