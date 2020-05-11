import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy.optimize import minimize, bisect

from numpy.polynomial.legendre import leggauss
from typing import Dict, Callable

from pylce.estimator import LCEEstimator

import pylce

print(pylce)

LCE_model = LCEEstimator({'alpha': 2.8, 'beta': 1.9, 'omega':5})

# +
y0 = 1
y1 = 3

# LCE_model._prior_beta_density(0.8)
prior_pdf = LCE_model.get_prior_pdf()
posterior_pdf = LCE_model.get_posterior_pdf(y0, y1)

# sanity check integration function
LCE_model._integration_calc(prior_pdf) ## 1
LCE_model._integration_calc(posterior_pdf)
# LCE_model._expectation_calc(prior_pdf) 
# 2* ( LCE_model.alpha / (LCE_model.alpha + LCE_model.beta)  ) - 1
# -

plot_xs = np.arange(-0.99, 1.0, 0.01)
plot_prior = prior_pdf(plot_xs)
plot_posterior = posterior_pdf(plot_xs)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(plot_xs, plot_prior, label='prior')
ax.plot(plot_xs, plot_posterior, label='posterior')
ax.legend()
plt.show()

print("prior mean: %.3f" % (2 * LCE_model.alpha / (LCE_model.alpha + LCE_model.beta) - 1))
print("posterior mean: %.3f" % LCE_model.fit_mean_estimator(y0,y1))
print("posterior mode: %.3f" % LCE_model.fit_mle_estimator(y0,y1))

# +
# get posterior mean & mode estimate as a function of y0 and y1
LCE_model.alpha = 2.8
LCE_model.beta = 1.9
print("prior mean: %.3f" % (2 * LCE_model.alpha / (LCE_model.alpha + LCE_model.beta) - 1))

L = 20
step = .5
posterior_mean = np.empty([L, L])
posterior_mode = np.empty([L, L])
y0 = np.arange(-L*step/2 + 0.01, L*step/2, step)
y1 = np.arange(-L*step/2 + 0.01, L*step/2, step)
for i in range(L):
    for j in range(L):
        posterior_mean[i,j] = LCE_model.fit_mean_estimator(y0[i], y1[j])
        posterior_mode[i,j] = LCE_model.fit_mle_estimator(y0[i], y1[j])

# -

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
title = 'posterior mean'
ax.set_xticks([])
ax.set_yticks([])
sns.heatmap(posterior_mean, cmap = 'RdYlGn', ax = ax)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
title = 'posterior mode'
ax.set_xticks([])
ax.set_yticks([])
sns.heatmap(posterior_mode, cmap = 'RdYlGn', ax = ax)
plt.show()

# we will work on 
sns.palplot(sns.color_palette("Set2"))
y = sns.color_palette("Set2")
y.as_hex()


