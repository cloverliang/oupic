# +
import dendropy as dp
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from scipy import stats
from munch import Munch
from typing import Dict, Callable
from tqdm.auto import tqdm

from pylce.pic import PIC
from pylce.simulator_corr_evo import *
from pylce.simulator_corr_util import *
# -

# 0. coefficients
with open('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/data/brawand_coef_3.pickel', 'rb') as file:
    brawand_coef = pickle.load(file)
with open('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/data/finches_coef_2.pickel', 'rb') as file:
    finches_coef = pickle.load(file)
with open('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/data/acer_coef_2.pickel', 'rb') as file:
    acer_coef = pickle.load(file)

brawand_coef.keys()

alphabets = ['A', 'B', 'C', 'D']

#### visualize coefficient matrix for all lambda
# %matplotlib notebook

acer_coef.keys()

# +
fig = plt.figure(figsize=(10, 5))
cmap = 'coolwarm'
alpha = 0.5
size_contrast = 100

# meshgrid
# xx, yy = np.meshgrid(np.arange(nspecies-1), np.arange(nspecies))
for row_i, pic_coef in enumerate([finches_coef, brawand_coef]):
    
    species = pic_coef['kappa=1.00'].T.index
    nspecies = len(species)
    pic_coef = {k: pic_coef[k] for k in ('raw', 'kappa=16.00', 'kappa=4.00', 'kappa=1.00', 'kappa=0.00')}
    
    # enumerate kappa / ts_lambda and visualize coefficient matrix
    for col_i, coef_idx in enumerate(pic_coef):

        # Do not plot raw coefficients
        if( coef_idx == 'raw'):
            continue
        
        # extract kappa from index
        kappa = float(coef_idx.split('=')[1])

        # setup axis
        # ax_contrast = fig.add_subplot(1, len(pic_coef), i+1)
        frame_idx = row_i * (len(pic_coef)-1) + col_i
        ax_contrast = fig.add_subplot(2, len(pic_coef)-1, frame_idx)

        # meshgrid
        zz_coef = pic_coef[coef_idx].T # species * contrast / n * (n-1)
        x_ticks = np.arange(zz_coef.shape[1])
        y_ticks = np.arange(zz_coef.shape[0])
        xx, yy = np.meshgrid(x_ticks, y_ticks)
        
        if( col_i != 1 ):
            # size_factor = np.abs(zz_coef.iloc[0, nspecies-2])
            # size_factor = np.abs(zz_coef.iloc[nspecies-1, nspecies-2])
            size_factor = np.abs(zz_coef.iloc[0, 0])
            plt.yticks(np.arange(nspecies), [])
        else:
            # size_factor = 1
            size_factor = np.abs(zz_coef.iloc[0, 0])
            if(row_i == 0):
                species_style = 'italic'
            else: 
                species_style = 'normal'
            plt.yticks(np.arange(nspecies), species, style = species_style)
            ax_contrast.text(-0.4, 1.15, alphabets[row_i], transform=ax_contrast.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
        plt.xticks(np.arange(nspecies-1), np.arange(nspecies-1) + 1 )
        zz_contrast = np.abs(zz_coef)*size_contrast / size_factor
        contrast_color = zz_coef.apply(lambda x: [0 if y <= 0 else 1 for y in x])

        ax_contrast.scatter(xx, yy, zz_contrast, c = contrast_color.values, cmap=cmap, alpha = alpha)

        fig_idx = r"$\kappa=%.2f$" % kappa
        if(kappa == 0):
            fig_idx = fig_idx + ' (BM)'
        ax_contrast.set_title(fig_idx)

plt.tight_layout()
plt.show()
# plt.savefig('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/contrast_coefficients.pdf')

# +
fig = plt.figure(figsize=(10, 2.8))
cmap = 'coolwarm'
alpha = 0.5
size_contrast = 20

# meshgrid
# xx, yy = np.meshgrid(np.arange(nspecies-1), np.arange(nspecies))
for row_i, pic_coef in enumerate([acer_coef]):
    
    species = pic_coef['kappa=1.00'].T.index
    nspecies = len(species)
    pic_coef = {k: pic_coef[k] for k in ('kappa=16.00', 'kappa=4.00', 'kappa=1.00', 'kappa=0.00')}
    
    # enumerate kappa / ts_lambda and visualize coefficient matrix
    for col_i, coef_idx in enumerate(pic_coef):
        
        # extract kappa from index
        kappa = float(coef_idx.split('=')[1])

        # setup axis
        # ax_contrast = fig.add_subplot(1, len(pic_coef), i+1)
        frame_idx = row_i * (len(pic_coef)) + col_i + 1
        ax_contrast = fig.add_subplot(1, len(pic_coef), frame_idx)

        # meshgrid
        zz_coef = pic_coef[coef_idx].T # species * contrast / n * (n-1)
        x_ticks = np.arange(zz_coef.shape[1])
        y_ticks = np.arange(zz_coef.shape[0])
        xx, yy = np.meshgrid(x_ticks, y_ticks)
        
        size_factor = np.abs(zz_coef.iloc[0, 0])
        # plt.yticks(np.arange(nspecies), [])
        
        # plt.xticks(np.arange(nspecies-1), np.arange(nspecies-1) + 1 )
        zz_contrast = np.abs(zz_coef)*size_contrast / size_factor
        contrast_color = zz_coef.apply(lambda x: [0 if y <= 0 else 1 for y in x])

        ax_contrast.scatter(xx, yy, zz_contrast, c = contrast_color.values, cmap=cmap, alpha = alpha)

        fig_idx = r"$\kappa=%.2f$" % kappa
        if(kappa == 0):
            fig_idx = fig_idx + ' (BM)'
        ax_contrast.set_title(fig_idx)

plt.tight_layout()
plt.show()
plt.savefig('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/Supp_contrast_coefficients_acer.pdf')
# -















# I. type I error rates
with open('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/data/brawand_type_I_error.pickel', 'rb') as file:
    brawand_error = pickle.load(file)
with open('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/data/finches_type_I_error.pickel', 'rb') as file:
    finches_error = pickle.load(file)
with open('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/data/acer_type_I_error.pickel', 'rb') as file:
    acer_error = pickle.load(file)

#### visualize coefficient matrix for all lambda
# %matplotlib notebook

# +
# Plot error rates
fig = plt.figure(figsize=(6, 8))

ax_error_rate = fig.add_subplot(3, 1, 1)
sns.heatmap( pd.DataFrame(finches_error).T, annot = True, fmt = ".4f", linewidths=.5, cbar = False, cmap = 'Oranges', vmax = 0.15)
ax_error_rate.set_title(r'Finch Tree, $n=8$, $\alpha=0.05$')
ax_error_rate.text(-0.1, 1.15, alphabets[0], transform=ax_error_rate.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

ax_error_rate = fig.add_subplot(3, 1, 2)
sns.heatmap( pd.DataFrame(brawand_error).T, annot = True, fmt = ".4f", linewidths=.5, cbar = False, cmap = 'Oranges', vmax = 0.15)
ax_error_rate.set_title(r'Brawand Tree, $n=10$, $\alpha=0.05$')
ax_error_rate.text(-0.1, 1.15, alphabets[1], transform=ax_error_rate.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

ax_error_rate = fig.add_subplot(3, 1, 3)
sns.heatmap( pd.DataFrame(acer_error).T, annot = True, fmt = ".4f", linewidths=.5, cbar = False, cmap = 'Oranges', vmax = 0.15)
ax_error_rate.set_title(r'Acer Tree, $n=55$, $\alpha=0.05$')
ax_error_rate.text(-0.1, 1.15, alphabets[2], transform=ax_error_rate.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

plt.tight_layout()
plt.show()
plt.savefig('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/type_I_error_rates.pdf')
# -



# +
# II. power
# -

with open('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/data/finches_power.pickel', 'rb') as file:
    finches_power = pickle.load(file)
with open('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/data/brawand_power.pickel', 'rb') as file:
    brawand_power = pickle.load(file)    
with open('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/data/acer_power.pickel', 'rb') as file:
    acer_power = pickle.load(file)

#### visualize coefficient matrix for all lambda
# %matplotlib notebook

# +
# Plot error rates
fig = plt.figure(figsize=(10, 8))
cmap = 'Spectral'

ax_ts_gamma = fig.add_subplot(3,2,1)
sns.heatmap( pd.DataFrame(finches_power['gamma=0.50']).T,
           annot = True, fmt = ".3f", linewidth=.5, cbar=False, cmap = cmap, vmin = 0, vmax = 1)
ax_ts_gamma.set_title(r'Finch Tree, n = 8, $\gamma=0.50$')
ax_ts_gamma.text(-0.1, 1.15, 'A', transform=ax_ts_gamma.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

ax_ts_gamma = fig.add_subplot(3,2,3)
sns.heatmap( pd.DataFrame(brawand_power['gamma=0.50']).T,
           annot = True, fmt = ".3f", linewidth=.5, cbar=False, cmap = cmap, vmin = 0, vmax = 1)
ax_ts_gamma.set_title(r'Brawand Tree, n = 10, $\gamma=0.50$')
ax_ts_gamma.text(-0.1, 1.15, 'B', transform=ax_ts_gamma.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

ax_ts_gamma = fig.add_subplot(3,2,5)
sns.heatmap( pd.DataFrame(acer_power['gamma=0.50']).T,
           annot = True, fmt = ".3f", linewidth=.5, cbar=False, cmap = cmap, vmin = 0, vmax = 1)
ax_ts_gamma.set_title(r'Acer Tree, n = 55, $\gamma=0.50$')
ax_ts_gamma.text(-0.1, 1.15, 'C', transform=ax_ts_gamma.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

ax_ts_gamma = fig.add_subplot(3,2,2)
sns.heatmap( pd.DataFrame(brawand_power['gamma=0.25']).T,
           annot = True, fmt = ".3f", linewidth=.5, cbar=False, cmap = cmap, vmin = 0, vmax = 1)
ax_ts_gamma.set_title(r'Brawand Tree, n = 10, $\gamma=0.25$')
ax_ts_gamma.text(-0.1, 1.15, 'D', transform=ax_ts_gamma.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

ax_ts_gamma = fig.add_subplot(3,2,4)
sns.heatmap( pd.DataFrame(brawand_power['gamma=0.75']).T,
           annot = True, fmt = ".3f", linewidth=.5, cbar=False, cmap = cmap, vmin = 0, vmax = 1)
ax_ts_gamma.set_title(r'Brawand Tree, n = 10, $\gamma=0.75$')
ax_ts_gamma.text(-0.1, 1.15, 'E', transform=ax_ts_gamma.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

ax_ts_gamma = fig.add_subplot(3,2,6)
sns.heatmap( pd.DataFrame(brawand_power['gamma=0.90']).T,
           annot = True, fmt = ".3f", linewidth=.5, cbar=False, cmap = cmap, vmin = 0, vmax = 1)
ax_ts_gamma.set_title(r'Brawand Tree, n = 10, $\gamma=0.90$')
ax_ts_gamma.text(-0.1, 1.15, 'F', transform=ax_ts_gamma.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

plt.tight_layout()
plt.show()
# plt.savefig('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/powers.pdf')
# +
# Plot error rates
fig = plt.figure(figsize=(6, 8))
cmap = 'Oranges'

ax_ts_gamma = fig.add_subplot(3,1,1)
sns.heatmap( pd.DataFrame(finches_power['gamma=0.75']).T,
           annot = True, fmt = ".3f", linewidth=.5, cbar=False, cmap = cmap)
ax_ts_gamma.set_title(r'Finch Tree, n = 8, $\gamma=0.75$')
ax_ts_gamma.text(-0.1, 1.15, 'A', transform=ax_ts_gamma.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

ax_ts_gamma = fig.add_subplot(3,1,2)
sns.heatmap( pd.DataFrame(brawand_power['gamma=0.75']).T,
           annot = True, fmt = ".3f", linewidth=.5, cbar=False, cmap = cmap)
ax_ts_gamma.set_title(r'Brawand Tree, n = 10, $\gamma=0.75$')
ax_ts_gamma.text(-0.1, 1.15, 'B', transform=ax_ts_gamma.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

ax_ts_gamma = fig.add_subplot(3,1,3)
sns.heatmap( pd.DataFrame(acer_power['gamma=0.75']).T,
           annot = True, fmt = ".3f", linewidth=.5, cbar=False, cmap = cmap)
ax_ts_gamma.set_title(r'Acer Tree, n = 55, $\gamma=0.75$')
ax_ts_gamma.text(-0.1, 1.15, 'C', transform=ax_ts_gamma.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

plt.tight_layout()
plt.show()
plt.savefig('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/Supp_powers_0.75.pdf')
# -

kappa = 4
kappa_hat = (32, 16, 8, 4, 2, 1, 0) 
gamma = 0.5
ROC curves (each tree add a curve)



# II. mse
with open('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/data/brawand_mse.pickel', 'rb') as file:
    brawand_mse = pickle.load(file)
with open('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/data/finches_mse.pickel', 'rb') as file:
    finches_mse = pickle.load(file)
with open('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/data/acer_mse.pickel', 'rb') as file:
    acer_mse = pickle.load(file)

#### visualize coefficient matrix for all lambda
# %matplotlib notebook

# +
# brawand_mse

# +
# Plot error rates
fig = plt.figure(figsize=(6, 8))
cmap = 'Oranges'

ax_ts_gamma = fig.add_subplot(3,1,1)
sns.heatmap( pd.DataFrame(brawand_mse['gamma=0.50']).T,
           annot = True, fmt = ".4f", linewidth=.5, cbar=False, cmap = cmap)
ax_ts_gamma.set_title(r'Finch Tree, n = 8, $\gamma=0.50$')
ax_ts_gamma.text(-0.1, 1.15, 'A', transform=ax_ts_gamma.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

ax_ts_gamma = fig.add_subplot(3,1,2)
sns.heatmap( pd.DataFrame(finches_mse['gamma=0.50']).T,
           annot = True, fmt = ".4f", linewidth=.5, cbar=False, cmap = cmap)
ax_ts_gamma.set_title(r'Brawand Tree, n = 10, $\gamma=0.50$')
ax_ts_gamma.text(-0.1, 1.15, 'B', transform=ax_ts_gamma.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

ax_ts_gamma = fig.add_subplot(3,1,3)
sns.heatmap( pd.DataFrame(acer_mse['gamma=0.50']).T,
           annot = True, fmt = ".4f", linewidth=.5, cbar=False, cmap = cmap)
ax_ts_gamma.set_title(r'Acer Tree, n = 55, $\gamma=0.50$')
ax_ts_gamma.text(-0.1, 1.15, 'C', transform=ax_ts_gamma.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

plt.tight_layout()
plt.show()
plt.savefig('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/mse.pdf')
# -










# +
# III. ROC curves
# -

with open('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/data/brawand_ROC.pickel', 'rb') as file:
    brawand_ROC = pickle.load(file)    
with open('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/data/acer_ROC.pickel', 'rb') as file:
    acer_ROC = pickle.load(file)    

#### visualize coefficient matrix for all lambda
# %matplotlib notebook

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for key in brawand_ROC['gamma=0.50']['$\\kappa=4$'].keys():
    kappa_hat = key
    if( kappa_hat in ['$\widehat{\kappa}=32$','$\widehat{\kappa}=8$','$\widehat{\kappa}=4$','$\widehat{\kappa}=1$','$\widehat{\kappa}=0$']):
        x = brawand_ROC['gamma=0.50']['$\\kappa=4$'][key]['X']
        y = brawand_ROC['gamma=0.50']['$\\kappa=4$'][key]['Y']
        ax.plot(x, y, '-', label = kappa_hat)
ax.legend()


