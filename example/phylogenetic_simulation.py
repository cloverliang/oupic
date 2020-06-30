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

from pylce.pic import PIC
from pylce.simulator_corr_evo import *
# -

# Different trees
# tree = dp.Tree.get(path = '/Users/cong/Documents/Projects/2020-per-gene-LCE/data/brawand_tree.nwk', schema = 'newick') # mammals, L = 320, n = 10
# tree = dp.Tree.get_from_path('/Users/cong/Documents/Projects/2020-per-gene-LCE/data/podos_tree.nwk', schema = 'newick') # finches, L = 5, n = 8
tree = dp.Tree.get_from_path('/Users/cong/Documents/Projects/2020-per-gene-LCE/data/acer_species_modified_branch.nwk', schema = 'newick') # acer, L = 60, n = 55
# tree = dp.Tree.get_from_path('/Users/cong/Documents/Projects/2020-per-gene-LCE/data/pinus_species.nwk', schema = 'newick') # pinus, L = 73, n = 111

# assign internal node label
idx = 1
for nd in tree.postorder_node_iter():
    if(nd.taxon == None): ## Add internal label name
        nd.label = "Internal_" + str(idx)
        idx = idx + 1
    else:
        nd.label = nd.taxon.label

# +
# tree.print_plot()
# -

L = 160
ts_lambda = 5/L
N = 5000
attr_xy = Munch( {'sigma_x':np.sqrt(10*ts_lambda), 'sigma_y':np.sqrt(10*ts_lambda), 'lambda_x':ts_lambda, 'lambda_y':ts_lambda, 'gamma_xy':.0, 'mu_x':3, 'mu_y':4} )
ts_lambda_series = np.array( [25/L, 5/L, 1/L, 0.2/L, 0.04/L] )
# gamma_xy_series = np.array( [0, .25, .5, .75, .9] )

# +
# Find coefficient matrix for all lambda
pic_coef = {}
species = [sp.label for sp in tree.taxon_namespace]

pic_coef['raw'] = pd.DataFrame( np.identity(len(species)), index = species, columns = species)

for ts_lambda in ts_lambda_series:
    idx = 'LT=' + str('{:.2f}'.format(ts_lambda*L))
    pic_coef[idx] = get_contrast_coef_ou(tree, {'ts_lambda':ts_lambda})[species]
    
pic_coef['bm'] = get_contrast_coef_bm(tree)[species]
# -

pic_coef.keys()

# simulate data, given lambda
x_rand, y_rand, sigma_xy = random_phylo_xy(tree, attr_xy, N)
x_rand = x_rand[species]
y_rand = y_rand[species]

# +
gamma_hat = { idx:[] for idx in pic_coef}
gamma_hat['raw'] = []

for idx in pic_coef:
    x_pic = x_rand.values @ pic_coef[idx].T.values
    y_pic = y_rand.values @ pic_coef[idx].T.values
    
    # find gamma_hat
    normalized_x_pic = normalize_vec(x_pic)
    normalized_y_pic = normalize_vec(y_pic)
    gamma_hat[idx] = np.einsum('ij,ij -> i', normalized_x_pic, normalized_y_pic)
# -

# format gamma_hat
gamma_hat_df = pd.DataFrame({'gamma_hat':[], 'type':[]})
for idx in gamma_hat:
    tmp = pd.DataFrame({'gamma_hat':gamma_hat[idx], 'type':np.repeat(idx, len(gamma_hat[idx]))})
    gamma_hat_df = gamma_hat_df.append(tmp, ignore_index=True)

# %matplotlib notebook

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 1, 1)
ax = sns.violinplot(x='type', y='gamma_hat', data=gamma_hat_df, palette="Set3")
plt.plot([-.5,6.5], [attr_xy.gamma_xy, attr_xy.gamma_xy], linestyle = '--')
ax.set_title('gamma=' + str(attr_xy.gamma_xy) + ' TL=' + str(attr_xy.lambda_x*L))
plt.show()

pd.DataFrame( gamma_hat ).describe()

stats.describe( pd.DataFrame(gamma_hat) )


