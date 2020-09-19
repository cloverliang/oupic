# +
import dendropy as dp
import numpy as np
import pandas as pd
import yaml
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import auc
from scipy import stats
from munch import Munch
from typing import Dict, Callable
from tqdm.auto import tqdm

from pylce.pic import PIC
from pylce.simulator_corr_evo import *
from pylce.simulator_corr_util import *

# +
# Different trees
# tree = dp.Tree.get(path = '/Users/cong/Documents/Projects/2020-per-gene-LCE/data/brawand_tree_reorder.nwk', schema = 'newick') # mammals, L = 320, n = 10
tree = dp.Tree.get_from_path('/Users/cong/Documents/Projects/2020-per-gene-LCE/data/podos_tree.nwk', schema = 'newick') # finches, L = 5, n = 8
# tree = dp.Tree.get_from_path('/Users/cong/Documents/Projects/2020-per-gene-LCE/data/acer_species_modified.nwk', schema = 'newick') # acer, L = 60, n = 55
# tree = dp.Tree.get_from_path('/Users/cong/Documents/Projects/2020-per-gene-LCE/data/pinus_species.nwk', schema = 'newick') # pinus, L = 73, n = 111
# tree = dp.Tree.get_from_path('/Users/cong/Documents/Projects/2020-per-gene-LCE/data/complete_binary_32.nwk', schema = 'newick') # fake, L = 104, n = 16

L = 5
tree_name = 'Finch'
# -

# assign internal node label
idx = 1
for nd in tree.postorder_node_iter():
    if(nd.taxon == None): ## Add internal label name
        nd.label = "Internal_" + str(idx)
        idx = idx + 1
    else:
        nd.label = nd.taxon.label

tree.print_plot()

# tree information
# species = [sp.label for sp in tree.taxon_namespace]
species = get_species_postorder(tree)
nspecies = len(species)

# +
# clear memory
# sol = {}
# ROC = {}
# -

# kappa_pairs_series = [[4.5, 5.5], [4,6], [3,7], [1,9]]
# kappa_pairs_series = [[9, 11], [8,12], [10,15]]
kappa_pairs_series = [[4.5, 5.5], [4,6], [3,7]]
N = 10000

# test different kappa pairs
for kappa_pairs in tqdm(kappa_pairs_series):
    # Define parameters
    kappa_x, kappa_y = kappa_pairs
    ts_lambda_x = get_lambda_from_kappa(kappa_x, L)
    ts_lambda_y = get_lambda_from_kappa(kappa_y, L)

    for kappa_hat in tqdm([kappa_x, kappa_y, (kappa_x + kappa_y)/2, 0]):
        ts_lambda_hat = get_lambda_from_kappa(kappa_hat, L)
        kappa_hat_idx = r"$\widehat{\kappa}=%.2f$" % kappa_hat

        # row idx
        row_idx = (tree_name + ' Tree', 
                   r"$\kappa_x={:.2f}$".format(kappa_x),
                   r"$\kappa_y={:.2f}$".format(kappa_y),
                   r"$\widehat{\kappa}=%.2f$" % kappa_hat)
        sol[row_idx] = {}
        ROC[row_idx] = {}
        
        # find pic coef
        if kappa_hat == 0:
            pic_coef = get_contrast_coef_bm(tree)
        else:
            pic_coef = get_contrast_coef_ou(tree, {'ts_lambda':ts_lambda_hat})

        # H0
        attr_xy_H0 = Munch( {'sigma_x':np.sqrt(10*ts_lambda_x), 
                          'sigma_y':np.sqrt(10*ts_lambda_y), 
                          'lambda_x':ts_lambda_x, 
                          'lambda_y':ts_lambda_y, 
                          'gamma_xy':0, 
                          'mu_x':0,
                          'mu_y':0,
                          'mode':'OU'} )
        x_rand_H0, y_rand_H0, sigma_xy = random_phylo_xy(tree, attr_xy_H0, N)
        x_rand_H0 = x_rand_H0[species]
        y_rand_H0 = y_rand_H0[species]

        # gamma_hat
        x_pic_H0 = x_rand_H0.values @ pic_coef.T.values
        y_pic_H0 = y_rand_H0.values @ pic_coef.T.values
        gamma_hat_H0 = calc_row_correlation(x_pic_H0, y_pic_H0)

        # Type I error (two sided)
        ts_alpha = 0.05
        ll_r, up_r = get_r_cutoff_contrast(nspecies, ts_alpha)
        sol[row_idx][( '','Type I error rate')] = ( (gamma_hat_H0 < ll_r).sum() + (gamma_hat_H0 > up_r).sum() ) / N

        # H1
        ts_gamma_series = [.25, .5, .75]
        for ts_gamma in ts_gamma_series:
            attr_xy_H1 = Munch( {'sigma_x':np.sqrt(10*ts_lambda_x), 
                              'sigma_y':np.sqrt(10*ts_lambda_y), 
                              'lambda_x':ts_lambda_x, 
                              'lambda_y':ts_lambda_y, 
                              'gamma_xy':ts_gamma, 
                              'mu_x':0,
                              'mu_y':0,
                              'mode':'OU'} )
            x_rand_H1, y_rand_H1, sigma_xy = random_phylo_xy(tree, attr_xy_H1, N)
            x_rand_H1 = x_rand_H1[species]
            y_rand_H1 = y_rand_H1[species]

            # ou gamma_hat
            x_pic_H1 = x_rand_H1.values @ pic_coef.T.values
            y_pic_H1 = y_rand_H1.values @ pic_coef.T.values
            gamma_hat_H1 = calc_row_correlation(x_pic_H1, y_pic_H1)

            ts_gamma_idx = r'$\gamma={:.2f}$'.format(ts_gamma)

            # power of test
            ts_threshold = np.quantile(gamma_hat_H0, 1 - ts_alpha)
            sol[row_idx][(ts_gamma_idx, 'power(\alpha=0.05)')] = (gamma_hat_H1 > ts_threshold).sum() / N
            # ROC curve
            ROC[row_idx][ts_gamma_idx] = {}
            false_positive = np.arange(201)/200
            quantiles = np.quantile(gamma_hat_H0, 1 - false_positive)
            true_positive = [(gamma_hat_H1 > quantile).sum()/N for quantile in quantiles]
            ROC[row_idx][ts_gamma_idx]['X'] = false_positive
            ROC[row_idx][ts_gamma_idx]['Y'] = true_positive
            
            # area under curve
            sol[row_idx][(ts_gamma_idx, 'auc')] = auc(false_positive, true_positive)
            
            # mean
            sol[row_idx][(ts_gamma_idx, 'mean')] = np.mean(gamma_hat_H1)
            # std
            sol[row_idx][(ts_gamma_idx, 'std')] = np.std(gamma_hat_H1)
            # mse
            sol[row_idx][(ts_gamma_idx, 'mse')] = np.mean( (gamma_hat_H1 - ts_gamma)**2 )



#### visualize coefficient matrix for all lambda
# %matplotlib notebook

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for key_level1 in ROC.keys():
    tree_name, kappa_x, kappa_y, kappa_hat = key_level1
    if(tree_name == 'Finch Tree'):
        ax_label = 0
    elif(tree_name == 'Brawand Tree'):
        ax_label = 1
    else:
        ax_label = 2
    for key_level2 in ROC[key_level1].keys():
        gamma = key_level2
        if( (kappa_x == '$\kappa_x=4.00$') & (kappa_y == '$\kappa_y=6.00$')):
            x = ROC[key_level1][key_level2]['X']
            y = ROC[key_level1][key_level2]['Y']
            if( kappa_hat == '$\widehat{\kappa}=0.00$' ):
                ax[ax_label].plot(x, y, ':', label = kappa_hat+";"+gamma)
            if( kappa_hat == '$\widehat{\kappa}=5.00$' ):
                ax[ax_label].plot(x, y, '-', label = kappa_hat+";"+gamma)
ax[0].legend()
ax[0].set_title('Finch Tree')
ax[1].legend()
ax[1].set_title('Brawand Tree')
ax[2].legend()
ax[2].set_title('Acer Tree')
plt.tight_layout()
plt.show()
plt.savefig('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/roc_curves.pdf')

pd.DataFrame(sol).T

# +
# pd.DataFrame(sol).T.to_csv('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/2020.08.06/unequal_lambda.csv')
# -









# 4. GLS with error term defined by kappa_x
y ~ x + epsilon

# 5. GLS with error term defined by kappa_y





