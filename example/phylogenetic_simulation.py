# +
import dendropy as dp
import numpy as np
import pandas as pd
import yaml
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

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
tree_name = 'finches'
# -

brawand_names = {'gga':'Chicken', 
                 'oan':'Platypus', 
                 'mdo':'Opossum', 
                 'mmu':'Mouse',
                 'ppy':'Orangutan',
                'ggo':'Gorilla',
                'hsa':'Human',
                'ppa':'Bonobo',
                'ptr':'Champanzee',
                'mml':'Macaque'}
for taxon in tree.taxon_namespace:
    taxon.label = brawand_names[taxon.label]

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

species

# series for L*T; ts_lambda; gamma
# lt_series = np.array([10,3.3,1,0])
k_series = np.array([32, 16, 8, 4, 2, 1, 0])
# k_series = np.array([20, 10, 5, 1, 0])
# k_series = np.array([100, 10, 5, 1, 0.1, 0])
# k_series = np.array([33, 10, 3.3, 1, 0])
ts_lambda_series = get_lambda_from_kappa(k_series, L)
ts_gamma_series = np.array( [.25, .5, .75, .9] )
# ts_gamma_series = np.array( [.5] )

# +
# Find coefficient matrix for all lambda
pic_coef = {}
pic_coef['raw'] = pd.DataFrame( np.identity(len(species)), index = species, columns = species)

for kappa in tqdm(k_series):
    idx = 'kappa=' + str('{:.2f}'.format(kappa))
    if kappa == 0:
        # pic_coef[idx + ' (BM)'] = get_contrast_coef_bm(tree)[species]
        pic_coef[idx] = get_contrast_coef_bm(tree)[species]
    else:
        ts_lambda = get_lambda_from_kappa(kappa, L)
        pic_coef[idx] = get_contrast_coef_ou(tree, {'ts_lambda': ts_lambda})[species]

print(pic_coef.keys())

# +
# with open('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/data/'+tree_name+'_coef_2.pickel', 'wb') as file:
#     documents = pickle.dump(pic_coef, file)
# -

#### visualize coefficient matrix for all lambda
# %matplotlib notebook

# +
fig = plt.figure(figsize=(15, 3))
cmap = 'coolwarm'
alpha = 0.5
size_contrast = 100

# meshgrid
# xx, yy = np.meshgrid(np.arange(nspecies-1), np.arange(nspecies))

# enumerate L*T / ts_lambda and visualize coefficient matrix
for i, idx in enumerate(pic_coef):

    if( idx == 'raw'):
        continue
    
    kappa = float(idx.split('=')[1])
    
    # setup axis
    # ax_contrast = fig.add_subplot(1, len(pic_coef), i+1)
    ax_contrast = fig.add_subplot(1, len(pic_coef)-1, i)
    
    # meshgrid
    zz_coef = pic_coef[idx].T # species * contrast / n * (n-1)
    x_ticks = np.arange(zz_coef.shape[1])
    y_ticks = np.arange(zz_coef.shape[0])
    xx, yy = np.meshgrid(x_ticks, y_ticks)
    
    if( i != 1 ):
        # size_factor = np.abs(zz_coef.iloc[0, nspecies-2])
        # size_factor = np.abs(zz_coef.iloc[nspecies-1, nspecies-2])
        size_factor = np.abs(zz_coef.iloc[0, 0])
        plt.yticks(np.arange(nspecies), [])
    else:
        # size_factor = 1
        size_factor = np.abs(zz_coef.iloc[0, 0])
        plt.yticks(np.arange(nspecies), species, style='italic')
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
# plt.savefig('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/'+tree_name+'_contrast_coef_3.pdf')
# -







# +
## I - type I error rates
# find theoretical cutoff
ts_alpha = 0.05
ll_r, up_r = get_r_cutoff_contrast(nspecies, ts_alpha)
ll_r_raw, up_r_raw = get_r_cutoff_tips(nspecies, ts_alpha)
lower_thresholds = np.concatenate( ([ll_r_raw], np.repeat(ll_r, len(pic_coef)-1 )) )
upper_thresholds = np.concatenate( ([up_r_raw], np.repeat(up_r, len(pic_coef)-1 )) )

# estimate type I error rates from simulated data; 
# also, power for 
N = 100000
error_rates = {}

for kappa in tqdm(k_series):
    # part I: type I error
    # set pars
    ts_lambda = get_lambda_from_kappa(kappa, L)
    if kappa == 0:
        attr_xy = Munch({'sigma_x': 10/np.sqrt(L), 
                          'sigma_y': 10/np.sqrt(L),
                          'gamma_xy':.0, 
                          'mu_x':0,
                          'mu_y':0,
                          'mode':'BM'} ) 
    else:
        attr_xy = Munch( {'sigma_x':np.sqrt(10*ts_lambda), 
                          'sigma_y':np.sqrt(10*ts_lambda), 
                          'lambda_x':ts_lambda, 
                          'lambda_y':ts_lambda, 
                          'gamma_xy':.0, 
                          'mu_x':0,
                          'mu_y':0,
                          'mode':'OU'} )
    
    # simulate data
    x_rand, y_rand, sigma_xy = random_phylo_xy(tree, attr_xy, N)
    x_rand = x_rand[species]
    y_rand = y_rand[species]
    
    gamma_hat = { }
    
    # pic for all lambda
    for idx in pic_coef:
        x_pic = x_rand.values @ pic_coef[idx].T.values
        y_pic = y_rand.values @ pic_coef[idx].T.values
        
        # find gamma_hat using x_pic and y_pic
        if(idx == 'raw'):
            kappa_hat_idx = "Raw Tips"
        else:
            kappa_hat = float(idx.split('=')[1])
            kappa_hat_idx = r"$\widehat{\kappa}=%.0f$" % kappa_hat
        gamma_hat[kappa_hat_idx] = calc_row_correlation(x_pic, y_pic)
        
    # find out error rates
    kappa_idx = r"$\kappa=%.0f$" % kappa
    error_rates[kappa_idx] = ( (pd.DataFrame(gamma_hat) < lower_thresholds).sum() + (pd.DataFrame(gamma_hat) > upper_thresholds).sum() )/ N

# +
# pd.DataFrame( error_rates ).T.to_csv('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/'+tree_name+'_type_I_error.csv')
# pd.DataFrame( error_rates ).T.style.format('{0:,.4f}').background_gradient(cmap='Oranges', axis=1)

# +
# with open('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/data/'+tree_name+'_type_I_error.pickel', 'wb') as file:
#     documents = pickle.dump(error_rates, file)
# -










ts_gamma_series


# +
# II. power of test 
ts_alpha = 0.05 #alpha
N = 10000
power = {}
ROC = {}

for ts_gamma in tqdm(ts_gamma_series):
    gm_name = 'gamma=' + str('{:.2f}'.format(ts_gamma))
    power[gm_name] = {}
    ROC[gm_name] = {}
    
    for kappa in k_series:
        # simulate H0 (gamma=0), set attributes
        if kappa == 0:
            # continue
            attr_xy = Munch({'sigma_x': 10/np.sqrt(L), 
                              'sigma_y': 10/np.sqrt(L),
                              'gamma_xy':.0, 
                              'mu_x':0,
                              'mu_y':0,
                              'mode':'BM'} )
            kappa_name = r"$\kappa=%.0f$" % kappa
        else:
            ts_lambda = get_lambda_from_kappa(kappa, L)
            attr_xy = Munch( {'sigma_x':np.sqrt(10*ts_lambda), 
                                  'sigma_y':np.sqrt(10*ts_lambda), 
                                  'lambda_x':ts_lambda, 
                                  'lambda_y':ts_lambda, 
                                  'gamma_xy':0, 
                                  'mu_x':0,
                                  'mu_y':0,
                                  'mode':'OU'} )
            kappa_name = r"$\kappa=%.0f$" % kappa

        # simulate data with ts_gamma
        x_rand_h0, y_rand_h0, sigma_xy_h0 = random_phylo_xy(tree, attr_xy, N)
        x_rand_h0 = x_rand_h0[species]
        y_rand_h0 = y_rand_h0[species]
    
        # simulate varied H1 (gamma!=0)
        # set H1 attributes
        attr_xy['gamma_xy'] = ts_gamma
        power[gm_name][kappa_name] = {}
        ROC[gm_name][kappa_name] = {}
        
        # simulate H1
        x_rand_h1, y_rand_h1, sigma_xy_h1 = random_phylo_xy(tree, attr_xy, N)
        x_rand_h1 = x_rand_h1[species]
        y_rand_h1 = y_rand_h1[species]
        
        # Find power with different assumed lambda
        for kappa_hat in k_series:
            kappa_hat_name = r"$\widehat{\kappa}=%.0f$" % kappa_hat
            idx = r"kappa=%.2f" % kappa_hat
            # 'LT=' + str('{:.2f}'.format(lt_hat))
            
            x_pic_h0 = x_rand_h0.values @ pic_coef[idx].T.values
            y_pic_h0 = y_rand_h0.values @ pic_coef[idx].T.values
            x_pic_h1 = x_rand_h1.values @ pic_coef[idx].T.values
            y_pic_h1 = y_rand_h1.values @ pic_coef[idx].T.values

            gamma_hat_h0 = calc_row_correlation(x_pic_h0, y_pic_h0)
            gamma_hat_h1 = calc_row_correlation(x_pic_h1, y_pic_h1)
            gamma_threshold = np.quantile(gamma_hat_h0, 1-ts_alpha)
            power[gm_name][kappa_name][kappa_hat_name] = np.sum(gamma_hat_h1 > gamma_threshold) / N
            
            # ROC
            ROC[gm_name][kappa_name][kappa_hat_name] = {}
            false_positive = np.arange(201)/200
            quantiles = np.quantile(gamma_hat_h0, 1 - false_positive)
            true_positive = [(gamma_hat_h1 > quantile).sum()/N for quantile in quantiles]
            ROC[gm_name][kappa_name][kappa_hat_name]['X'] = false_positive
            ROC[gm_name][kappa_name][kappa_hat_name]['Y'] = true_positive


# +
# pd.DataFrame(power['gamma=0.25']).T.style.format('{0:,.4f}').background_gradient(cmap='Oranges', axis=1)
# pd.DataFrame(power['gamma=0.90']).T.style.format('{0:,.4f}').background_gradient(cmap='Oranges', axis=1)

# +
# pd.DataFrame( power ).T.to_csv('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/' + tree_name + '_power.csv')
# -

# with open('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/data/'+tree_name+'_power.pickel', 'wb') as file:
#     documents = pickle.dump(power, file)
with open('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/data/'+tree_name+'_ROC.pickel', 'wb') as file:
    documents = pickle.dump(ROC, file)

#### visualize coefficient matrix for all lambda
# %matplotlib notebook

# +
# Plot error rates
fig = plt.figure(figsize=(9, 5))

for i, ts_gamma in enumerate(ts_gamma_series):
    ax_ts_gamma = fig.add_subplot(2,2,i+1)
    gamma_idx = 'gamma=' + '{:.2f}'.format(ts_gamma)
    sns.heatmap( pd.DataFrame(power[gamma_idx]).T,
               annot = True, fmt = ".4f", linewidth=.5, cbar=False, cmap = 'Oranges')
    gamma_plot_idx = r'$\gamma_{{xy}}={:.2f}$'.format(ts_gamma)
    ax_ts_gamma.set_title(gamma_plot_idx)

plt.tight_layout()
plt.show()
# -











# +
# III. Bias, Variance and MSE
ts_alpha = 0.05 #alpha
N = 10000
res = {}
# ts_gamma_series = np.array( [0, .25, .5, .75, .9] )

for ts_gamma in tqdm(ts_gamma_series):
    gm_name = 'gamma=' + str('{:.2f}'.format(ts_gamma))
    res[gm_name] = {}
    
    for kappa in k_series:
        # simulate H0 (gamma=0), set attributes
        if kappa == 0:
            # continue
            attr_xy = Munch({'sigma_x': 10/np.sqrt(L), 
                              'sigma_y': 10/np.sqrt(L),
                              'gamma_xy': ts_gamma, 
                              'mu_x':0,
                              'mu_y':0,
                              'mode':'BM'} )
            kappa_name = r"$\kappa=%.0f$" % kappa
        else:
            ts_lambda = get_lambda_from_kappa(kappa, L)
            attr_xy = Munch( {'sigma_x':np.sqrt(10*ts_lambda), 
                                  'sigma_y':np.sqrt(10*ts_lambda), 
                                  'lambda_x':ts_lambda, 
                                  'lambda_y':ts_lambda, 
                                  'gamma_xy': ts_gamma, 
                                  'mu_x':0,
                                  'mu_y':0,
                                  'mode':'OU'} )
            kappa_name = r"$\kappa=%.0f$" % kappa
        res[gm_name][kappa_name] = {}
        
        # simulate data with ts_gamma
        x_rand, y_rand, sigma_xy = random_phylo_xy(tree, attr_xy, N)
        x_rand = x_rand[species]
        y_rand = y_rand[species]
    
        # Find variance, bias, mse with different lambda hat
        for kappa_hat in k_series:
            idx = r"kappa=%.2f" % kappa_hat
            kappa_hat_name = r"$\widehat{\kappa}=%.0f$" % kappa_hat
            
            x_pic = x_rand.values @ pic_coef[idx].T.values
            y_pic = y_rand.values @ pic_coef[idx].T.values
            gamma_hat = calc_row_correlation(x_pic, y_pic)
            # res[gm_name][kappa_name][kappa_hat_name + ' var'] = np.var(gamma_hat)
            # res[gm_name][kappa_name][kappa_hat_name + ' bias'] = (np.mean(gamma_hat) - ts_gamma)**2
            res[gm_name][kappa_name][kappa_hat_name] = np.sum( (gamma_hat - ts_gamma)**2 ) / N

# +
# pd.DataFrame( res['gamma=0.90'] ).T.filter(like = 'mse').style.background_gradient(cmap='Oranges', axis=1)
# pd.DataFrame( res['gamma=0.50'] ).T.style.background_gradient(cmap='Oranges', axis=1)
# -

tree_name

with open('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/data/'+tree_name+'_mse.pickel', 'wb') as file:
    documents = pickle.dump(res, file)

#### visualize coefficient matrix for all lambda
# %matplotlib notebook

ts_gamma_series

# +
# Plot error rates
fig = plt.figure(figsize=(10, 6))

for i, ts_gamma in enumerate(ts_gamma_series):
    ax_ts_gamma = fig.add_subplot(2,2,i+1)
    gamma_idx = 'gamma=' + '{:.2f}'.format(ts_gamma)
    sns.heatmap( pd.DataFrame(res[gamma_idx]).T,
               annot = True, fmt = ".4f", linewidth=.5, cbar=False, cmap = 'Oranges')
    gamma_plot_idx = r'$\gamma_{{xy}}={:.2f}$'.format(ts_gamma)
    ax_ts_gamma.set_title(gamma_plot_idx)

plt.tight_layout()
plt.show()
# -


