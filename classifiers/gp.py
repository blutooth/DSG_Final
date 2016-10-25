print(__doc__)

# Author: Vincent Dubourg <vincent.dubourg@gmail.com>
# Licence: BSD 3 clause

import numpy as np
from scipy import stats
from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as pl
from matplotlib import cm



# Standard normal distribution functions
phi = stats.distributions.norm().pdf
PHI = stats.distributions.norm().cdf
PHIinv = stats.distributions.norm().ppf

# A few constants
lim = 8


def g(x):
    """The function to predict (classification will then consist in predicting
    whether g(x) <= 0 or not)"""
    return 5. - x[:, 1] - .5 * x[:, 0] ** 2.

# Design of experiments
X = np.array([[-4.61611719, -6.00099547],
              [4.10469096, 5.32782448],
              [0.00000000, -0.50000000],
              [-6.17289014, -4.6984743],
              [1.3109306, -6.93271427],
              [-5.03823144, 3.10584743],
              [-2.87600388, 6.74310541],
              [5.21301203, 4.26386883]])

# Observations
y = g(X)

# Instanciate and fit Gaussian Process Model
gp = GaussianProcess(theta0=5e-1)

# Don't perform MLE or you'll get a perfect prediction for this simple example!
gp.fit(X, y)

# Evaluate real function, the prediction and its MSE on a grid
res = 50
x1, x2 = np.meshgrid(np.linspace(- lim, lim, res),
                     np.linspace(- lim, lim, res))
xx = np.vstack([x1.reshape(x1.size), x2.reshape(x2.size)]).T

y_true = g(xx)
y_pred, MSE = gp.predict(xx, eval_MSE=True)
sigma = np.sqrt(MSE)
y_true = y_true.reshape((res, res))
y_pred = y_pred.reshape((res, res))
sigma = sigma.reshape((res, res))
k = PHIinv(.975)

# Plot the probabilistic classification iso-values using the Gaussian property
# of the prediction
fig = pl.figure(1)
ax = fig.add_subplot(111)
ax.axes.set_aspect('equal')
pl.xticks([])
pl.yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
pl.xlabel('$x_1$')
pl.ylabel('$x_2$')

cax = pl.imshow(np.flipud(PHI(- y_pred / sigma)), cmap=cm.gray_r, alpha=0.8,
                extent=(- lim, lim, - lim, lim))
norm = pl.matplotlib.colors.Normalize(vmin=0., vmax=0.9)
cb = pl.colorbar(cax, ticks=[0., 0.2, 0.4, 0.6, 0.8, 1.], norm=norm)
cb.set_label('${\\rm \mathbb{P}}\left[\widehat{G}(\mathbf{x}) \leq 0\\right]$')

pl.plot(X[y <= 0, 0], X[y <= 0, 1], 'r.', markersize=12)

pl.plot(X[y > 0, 0], X[y > 0, 1], 'b.', markersize=12)

cs = pl.contour(x1, x2, y_true, [0.], colors='k', linestyles='dashdot')

cs = pl.contour(x1, x2, PHI(- y_pred / sigma), [0.025], colors='b',
                linestyles='solid')
pl.clabel(cs, fontsize=11)

cs = pl.contour(x1, x2, PHI(- y_pred / sigma), [0.5], colors='k',
                linestyles='dashed')
pl.clabel(cs, fontsize=11)

cs = pl.contour(x1, x2, PHI(- y_pred / sigma), [0.975], colors='r',
                linestyles='solid')
pl.clabel(cs, fontsize=11)

pl.show()



'''
import GPflow
import sys
import csv
import numpy as np
import GPflow
from sklearn.cross_validation import train_test_split
import time
from sklearn.preprocessing import normalize



def kernel():
   nDim = 1
   kern = GPflow.kernels.RBF(nDim) + GPflow.kernels.White(nDim)
   kern.white.variance = 0.1
   return kern

def likelihood():
   return GPflow.likelihoods.Bernoulli()

def toggleHypers(model,fixed):
   model.kern.rbf.lengthscales.fixed = fixed
   model.kern.rbf.variance.fixed = fixed
   model.kern.white.variance.fixed = fixed
   return model

def get_accuracy(model,X,Y):
   model.optimize(max_iters=max_iters)
   pred=[round(x) for y in m1.predict_y(X) for x in y]
   error=abs(pred-Y[0])
   time.sleep(10)
   accuracy=1-sum(error)/len(pred)
   return accuracy

# normalise dimension
#Setup the experiment and plotting.
Ms = [200] # 8 16 32
X = np.load('../Data/X.npy')

Y = np.reshape(X[:, 0],(-1,1))
X = normalize(X[:, 1:],norm='l1', axis=1)



Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,
   Y, test_size=0.2, random_state=42)


max_iters = 500

#Run sparse classification with increasing number of inducing points
for index, num_inducing in enumerate(Ms):
   print ('pseudo: ',Ms[index])
   #kmeans for selecting Z
   from scipy.cluster.vq import kmeans
   Z = kmeans(X, num_inducing)[0]
   m1 = GPflow.svgp.SVGP(Xtrain, Ytrain, kern=kernel(), likelihood=likelihood(), Z=Z )



   #Unfix the hyperparameters.
   toggleHypers( m1, False )
   m1.optimize(max_iters=max_iters)


   train=get_accuracy(m1,Xtrain,Ytrain)
   test=get_accuracy(m1,Xtest,Ytest)
   print('test: ',test)
   print('train: ',train)
'''

