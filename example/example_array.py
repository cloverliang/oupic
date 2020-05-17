# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# Import modules 
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import multivariate_normal as mvn
from scipy.optimize import minimize, bisect
from scipy.special import gamma, factorial

from numpy.polynomial.legendre import leggauss
from typing import Dict, Callable

# Note: import local modules last
from estimator_array import LCEEstimator_array
# -

LCE_model = LCEEstimator_array({'alpha': 2.79, 'beta': 1.65, 'omega': np.repeat(5, 9), 'nspecies':10})

# +
prior_pdf = LCE_model.get_prior_pdf()
ya = np.array( [1, 5, 3, 4, 5, 1, 0, 2, -4] )
yb = np.array( [3, 5, 4, -3, 4, 5, 2, 3, 3] )

print( "ncontrast: %.3f" % LCE_model.ncontrasts)
print( "check log likelihood: %.3f" % LCE_model.get_log_likelihood(0.99, ya, yb) )
print( "check log likelihood hardcode: %.3f" % LCE_model.get_log_likelihood_hardcode(0.99, ya, yb))
print( "check normal: %.20f" % LCE_model._normal_density(0.3, ya, yb)) # problematic
print( "check normal autoscale: %.20f" % LCE_model._normal_density_autoscale(0.3, ya, yb))

# LCE_model = LCEEstimator_array({'alpha': 2.79, 'beta': 1.65, 'omega': np.repeat(5, 3), 'nspecies':4})
# ya = np.array( [1,0.1,1] )
# yb = np.array( [2,3,-2] )

posterior_pdf = LCE_model.get_posterior_pdf(ya, yb )
prior_pdf = LCE_model.get_prior_pdf()

plot_xs = np.arange(-0.99, 1.0, 0.01)
plot_prior = prior_pdf(plot_xs)
plot_posterior = posterior_pdf(plot_xs)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(plot_xs, plot_prior, label='prior')
ax.plot(plot_xs, plot_posterior, label='posterior')
ax.axvline(LCE_model.fit_map_estimator(ya, yb), color = 'orange', ls = '--')
ax.legend()
plt.show()

print( "prior mean %.3f" % LCE_model.get_prior_mean() )
print( "posterior mean %.3f" % LCE_model.fit_mean_estimator(ya, yb) )
print( "Sanity check integration %.3f" % LCE_model._integration_calc(posterior_pdf))
print( "Sanity check expectation %.3f" % LCE_model._expectation_calc(posterior_pdf))
print( "posterior max %.3f" % LCE_model.fit_map_estimator(ya, yb) )
# -

# Next read the real data and plot the histogram of posterior estimations
## 1. read data from a tab deliminated file including gene expression contrast values from several species and tissue types
## 2. parse data 
## 3. get per-gene posterior estimation
## 4. plot distribution of posterior estimations
brain = pd.read_csv('/Users/cong/Documents/Projects/2020-per-gene-LCE/data/brawand_brain.txt', delimiter = '\t', index_col=0)
heart = pd.read_csv('/Users/cong/Documents/Projects/2020-per-gene-LCE/data/brawand_heart.txt', delimiter = '\t', index_col=0)

heart.var(axis = 0)
# brain.var(axis = 0)

prior_cor = []
for i in tqdm(range(heart.shape[0])):
     prior_cor.append( np.corrcoef(brain.iloc[i], heart.iloc[i])[1,0] )

LCE_model = LCEEstimator_array({'alpha': 2.79, 'beta': 1.65, 'omega': np.repeat(5, 9), 'nspecies':10})
rho_bar = np.nanmean(prior_cor)
rho_var = np.nanvar(prior_cor)
LCE_model.alpha = (1 - rho_bar) * ((1 + rho_bar) ** 2) / 2 / rho_var - (1 + rho_bar) / 2
LCE_model.beta = (1 + rho_bar) * ((1 - rho_bar) ** 2) / 2 / rho_var - (1 - rho_bar) / 2
print("Estimated alpha: %.3f" % LCE_model.alpha)
print("Estimated beta: %.3f" % LCE_model.beta)

# +
# LCE_model = LCEEstimator_array({'alpha': 2.72, 'beta': 1.61, 'omega': np.array([5,5,5,5,5,5,5,5,5]) *2 , 'nspecies':10})
LCE_model.omega = np.repeat(25, 9)
ya = brain.iloc[3]
yb = heart.iloc[3]

prior_pdf = LCE_model.get_prior_pdf()
posterior_pdf = LCE_model.get_posterior_pdf(ya, yb)

print( ya, yb)
print( LCE_model._normal_density(0.3, ya, yb) )
print( LCE_model._normal_density_autoscale(0.3, ya, yb) )
print( posterior_pdf(0.3))
print( posterior_pdf(0.05))
## auto scale could still be problematic since the probability at rho=0 was used to scale the density, 
## while we know that the posterior density could be 

# +
plot_xs = np.arange(-0.99, 1.0, 0.01)
plot_prior = prior_pdf(plot_xs)
plot_posterior = posterior_pdf(plot_xs)

fig, ax = plt.subplots(1, 2, figsize=(20, 6))
ax[0].hist(prior_cor, bins = 50, density = True)
ax[0].plot(plot_xs, plot_prior)

ax[1].plot(plot_xs, plot_prior, label='prior')
ax[1].plot(plot_xs, plot_posterior, label='posterior')
ax[1].axvline(LCE_model.fit_map_estimator(ya, yb), color = 'orange', ls = '--')
ax[1].legend()
plt.show()

print( "prior mean %.3f" % LCE_model.get_prior_mean() )
print( "posterior mean %.3f" % LCE_model.fit_mean_estimator(ya, yb) )
print( "Sanity check integration %.3f" % LCE_model._integration_calc(posterior_pdf))
print( "Sanity check expectation %.3f" % LCE_model._expectation_calc(posterior_pdf))
print( "posterior max %.3f" % LCE_model.fit_map_estimator(ya, yb) )
# -

posterior_mean = []
posterior_mle = []
for i in tqdm(range(heart.shape[0])):
    posterior_mean.append(LCE_model.fit_mean_estimator(brain.iloc[i], heart.iloc[i]))
    posterior_mle.append(LCE_model.fit_map_estimator(brain.iloc[i], heart.iloc[i]))

np.where( np.isnan(posterior_mean) )

np.where( np.isnan(posterior_mle) )

# +
## Plot section
plot_xs = np.arange(-0.99, 1.0, 0.01)
prior_pdf = LCE_model.get_prior_pdf()
plot_prior = prior_pdf(plot_xs) 

fig, ax = plt.subplots(1, 2, figsize=(20, 6))
ax[0].hist(posterior_mean, bins = 50, density = True)
ax[0].set_title('distribution of posterior mean estimation')
ax_pdf1 = ax[0].twinx() 
ax_pdf1.plot(plot_xs, plot_prior, label='prior', color = 'orange')

ax[1].hist(posterior_mle, bins = 50, density = True)
ax[1].set_title('distribution of maximum a posteriori estimation')
ax_pdf2 = ax[1].twinx() 
ax_pdf2.plot(plot_xs, plot_prior, label='prior', color = 'orange')

plt.show()
print( "alpha: %.3f" % LCE_model.alpha)
print( "beta: %.3f" % LCE_model.beta)
print( "Omega: %.1f" % np.mean(LCE_model.omega) )
print( "prior mean: %.3f" % LCE_model.get_prior_mean())
print( "average posterior mean estimation: %.3f" % np.nanmean( posterior_mean ) )
print( "average maximum a posteriori estimation: %.3f" % np.nanmean( posterior_mle ) )
# -

fig, ax = plt.subplots(1, 3, figsize=(25, 6))
ax[0].scatter(prior_cor, posterior_mean, s = 1)
ax[0].set_title("prior_cor vs posterior_mean")
ax[0].plot([-1,1], [-1,1], color = 'orange')
ax[1].scatter(prior_cor, posterior_mle, s = 1)
ax[1].set_title("prior_cor vs maximum_a_posteriori")
ax[1].plot([-1,1], [-1, 1], color = 'orange')
ax[2].scatter(posterior_mean, posterior_mle, s = 1)
ax[2].set_title("posterior_mean vs maximum_a_posteriori")
ax[2].plot([-1,1], [-1, 1], color = 'orange')
plt.show()


# +
## Double check autoscale function
## simulation result -> check the understanding of results, variance etc. when w is small, the influence of data on posterior estimation is stronger. 
## check off diagonal values 
