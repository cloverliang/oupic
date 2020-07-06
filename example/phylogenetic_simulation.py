# +
import dendropy as dp
import numpy as np
import pandas as pd
import yaml
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
# tree = dp.Tree.get_from_path('/Users/cong/Documents/Projects/2020-per-gene-LCE/data/podos_tree.nwk', schema = 'newick') # finches, L = 5, n = 8
# tree = dp.Tree.get_from_path('/Users/cong/Documents/Projects/2020-per-gene-LCE/data/acer_species_modified.nwk', schema = 'newick') # acer, L = 60, n = 55
tree = dp.Tree.get_from_path('/Users/cong/Documents/Projects/2020-per-gene-LCE/data/pinus_species.nwk', schema = 'newick') # pinus, L = 73, n = 111
# tree = dp.Tree.get_from_path('/Users/cong/Documents/Projects/2020-per-gene-LCE/data/complete_binary_32.nwk', schema = 'newick') # fake, L = 104, n = 16

L = 73
tree_name = 'pinus'
# -

# assign internal node label
idx = 1
for nd in tree.postorder_node_iter():
    if(nd.taxon == None): ## Add internal label name
        nd.label = "Internal_" + str(idx)
        idx = idx + 1
    else:
        nd.label = nd.taxon.label

# tree information
# species = [sp.label for sp in tree.taxon_namespace]
species = get_species_postorder(tree)
nspecies = len(species)

# series for L*T; ts_lambda; gamma
# lt_series = np.array([10,3.3,1,0])
lt_series = np.array([16, 8, 4, 2, 1, 0.5, 0])
ts_lambda_series = lt_series / L
ts_gamma_series = np.array( [.25, .5, .75, .9] )

# +
# Find coefficient matrix for all lambda
pic_coef = {}
pic_coef['raw'] = pd.DataFrame( np.identity(len(species)), index = species, columns = species)

for lt in tqdm(lt_series):
    idx = 'LT=' + str('{:.2f}'.format(lt))
    if lt == 0:
        pic_coef[idx + ' (BM)'] = get_contrast_coef_bm(tree)[species]
    else:
        pic_coef[idx] = get_contrast_coef_ou(tree, {'ts_lambda': lt/L})[species]

print(pic_coef.keys())
# -

#### visualize coefficient matrix for all lambda
# %matplotlib notebook

# +
fig = plt.figure(figsize=(14, 3))
cmap = 'coolwarm'
alpha = 0.5
size_contrast = 20

# meshgrid
# xx, yy = np.meshgrid(np.arange(nspecies-1), np.arange(nspecies))

# enumerate L*T / ts_lambda and visualize coefficient matrix
for i, idx in enumerate(pic_coef):

    # setup axis
    ax_contrast = fig.add_subplot(1, len(pic_coef), i+1)
    
    # meshgrid
    zz_coef = pic_coef[idx].T # species * contrast / n * (n-1)
    x_ticks = np.arange(zz_coef.shape[1])
    y_ticks = np.arange(zz_coef.shape[0])
    xx, yy = np.meshgrid(x_ticks, y_ticks)
    
    if( idx != 'raw'):
        size_factor = np.abs(zz_coef.iloc[0, nspecies-2])
        # size_factor = np.abs(zz_coef.iloc[nspecies-1, nspecies-2])
        plt.yticks(np.arange(nspecies), [])
    else:
        size_factor = 1
        # plt.yticks(np.arange(nspecies), species)

    zz_contrast = np.abs(zz_coef)*size_contrast / size_factor
    contrast_color = zz_coef.apply(lambda x: [0 if y <= 0 else 1 for y in x])
    
    ax_contrast.scatter(xx, yy, zz_contrast, c = contrast_color.values, cmap=cmap, alpha = alpha)
    ax_contrast.set_title(idx + ' coefficient')
        
plt.tight_layout()
plt.show()
# plt.savefig('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/'+tree_name+'_contrast_coef.pdf')
# -







# +
# find theoretical cutoff
ts_alpha = 0.05
ll_r, up_r = get_r_cutoff(nspecies, ts_alpha)

# estimate type I error rates from simulated data; 
# also, power for 
N = 10000
error_rates = {}

for lt in tqdm(lt_series):
    # part I: type I error
    # set pars
    ts_lambda = lt / L
    if lt == 0:
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
        gamma_hat[idx] = calc_row_correlation(x_pic, y_pic)
        
    # find out error rates
    lt_name = 'LT=' + str('{:.2f}'.format(lt))
    error_rates[lt_name] = ( (pd.DataFrame(gamma_hat) < ll_r).sum() + (pd.DataFrame(gamma_hat) > up_r).sum() )/ N
# -

# pd.DataFrame( error_rates ).T.to_csv('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/'+tree_name+'_type_I_error.csv')
pd.DataFrame( error_rates ).T.style.background_gradient(cmap='Oranges', axis=1)







# +
# find power with different ts_gamma
ts_alpha = 0.05 #alpha
N = 10000
power = {}

for lt in tqdm(lt_series):
    # simulate H0 (gamma=0), set attributes
    if lt == 0:
        # continue
        attr_xy = Munch({'sigma_x': 10/np.sqrt(L), 
                          'sigma_y': 10/np.sqrt(L),
                          'gamma_xy':.0, 
                          'mu_x':0,
                          'mu_y':0,
                          'mode':'BM'} )
        lt_name = 'LT=' + str('{:.2f}'.format(lt)) + ' (BM)'
    else:
        ts_lambda = lt / L
        attr_xy = Munch( {'sigma_x':np.sqrt(10*ts_lambda), 
                              'sigma_y':np.sqrt(10*ts_lambda), 
                              'lambda_x':ts_lambda, 
                              'lambda_y':ts_lambda, 
                              'gamma_xy':0, 
                              'mu_x':0,
                              'mu_y':0,
                              'mode':'OU'} )
        lt_name = 'LT=' + str('{:.2f}'.format(lt))
    power[lt_name] = {}
        
    # simulate data with ts_gamma
    x_rand_h0, y_rand_h0, sigma_xy_h0 = random_phylo_xy(tree, attr_xy, N)
    x_rand_h0 = x_rand_h0[species]
    y_rand_h0 = y_rand_h0[species]
    
    gamma_hat_h0 = {}
    # pic with true ts_gamma, and 
    x_pic_h0_ou = x_rand_h0.values @ pic_coef[lt_name].T.values
    y_pic_h0_ou = y_rand_h0.values @ pic_coef[lt_name].T.values
    x_pic_h0_bm = x_rand_h0.values @ pic_coef['LT=0.00 (BM)'].T.values    
    y_pic_h0_bm = y_rand_h0.values @ pic_coef['LT=0.00 (BM)'].T.values
    
    gamma_hat_h0['ou'] = calc_row_correlation(x_pic_h0_ou, y_pic_h0_ou)
    gamma_hat_h0['bm'] = calc_row_correlation(x_pic_h0_bm, y_pic_h0_bm)
    
    ou_threshold = np.quantile(gamma_hat_h0['ou'], 1-ts_alpha)
    bm_threshold = np.quantile(gamma_hat_h0['bm'], 1-ts_alpha)
    
    # simulate H1 (gamma!=0)
    for ts_gamma in ts_gamma_series:
        gm_name = 'gamma=' + str('{:.2f}'.format(ts_gamma))
        attr_xy['gamma_xy'] = ts_gamma
        
        # simulate data with ts_gamma
        x_rand_h1, y_rand_h1, sigma_xy_h1 = random_phylo_xy(tree, attr_xy, N)
        x_rand_h1 = x_rand_h1[species]
        y_rand_h1 = y_rand_h1[species]
        
        gamma_hat_h1 = {}
        
        x_pic_h1_ou = x_rand_h1.values @ pic_coef[lt_name].T.values
        y_pic_h1_ou = y_rand_h1.values @ pic_coef[lt_name].T.values
        x_pic_h1_bm = x_rand_h1.values @ pic_coef['LT=0.00 (BM)'].T.values    
        y_pic_h1_bm = y_rand_h1.values @ pic_coef['LT=0.00 (BM)'].T.values

        gamma_hat_h1['ou'] = calc_row_correlation(x_pic_h1_ou, y_pic_h1_ou)
        gamma_hat_h1['bm'] = calc_row_correlation(x_pic_h1_bm, y_pic_h1_bm)

        # power using true ou model
        power[lt_name]['ou_pic: ' + gm_name] = np.sum(gamma_hat_h1['ou'] > ou_threshold ) / N
        # power using BM model
        power[lt_name]['bm_pic: '+ gm_name] = np.sum(gamma_hat_h1['bm'] > bm_threshold ) / N
# -

# pd.DataFrame( power ).T.to_csv('/Users/cong/Documents/Projects/2020-per-gene-LCE/figures/' + tree_name + '_power.csv')
pd.DataFrame( power ).T


