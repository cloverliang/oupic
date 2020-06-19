# +
# import sys
# sys.path.insert(0, '/Users/cong/Workspace/repos/pylce')
# sys.path
# -

import pandas as pd
import numpy as np
import dendropy as dp
import matplotlib.pyplot as plt
from scipy import optimize
from typing import Dict, Callable
from scipy.spatial.distance import pdist, squareform
import yaml
import warnings
from pylce.other_pars_util import *

# read brawand data
brawand = pd.read_csv('/Users/cong/Documents/Projects/2020-per-gene-LCE/data/brawand_RPKM_ave_df.txt', delimiter = '\t', index_col=0)

# parse tissue and species information
ts = {}
sp = {}
for sample in brawand.columns:
    species,tissue = sample.split('_')
    if tissue not in ts:
        ts[tissue] = []
    ts[tissue].append(species)
    if species not in sp:
        sp[species] = []
    sp[species].append(tissue)
print(sp)
print(ts)

# read tree and map species name
tree = dp.Tree.get(path = '/Users/cong/Documents/Projects/2020-per-gene-LCE/data/brawand_tree.nwk', schema = 'newick')
with open('/Users/cong/Documents/Projects/2020-per-gene-LCE/data/taxa_name_abbr.yaml', 'r') as file:
    taxa_short_to_long = yaml.safe_load(file)
taxa_long_to_short = {v: k for k, v in taxa_short_to_long.items()}
for key in tree.taxon_namespace:
    key.label = taxa_long_to_short[key.label]
tree.print_plot()
print(tree.as_string(schema='newick'))

# %matplotlib notebook

# +
# parameters for each tissue type
sol = {}
# fig, ax = plt.subplots(2, 3, figsize=(10, 6))
fig = plt.figure(figsize=(10, 6))

for idx, tissue in enumerate(ts):
    ax = fig.add_subplot(2, 3, idx + 1)
    # expression data
    samples = [f'{species}_{tissue}' for species in ts[tissue]]
    exp_data = np.sqrt(brawand[samples])
    exp_data.columns = ts[tissue]

    # estimate parameters
    res = fit_sigma_lambda_estimator(exp_data, tree)
    
    # visualize
    xnew = np.arange(0, 650)
    ynew = phy_dist(xnew, res['OptimizedValue'][0], res['OptimizedValue'][1] )
    ax.scatter(res['xtrain'], res['ytrain'], marker = '.', alpha = 0.7)
    ax.plot(xnew, ynew, color = 'red', alpha = 0.7)
    ax.set_title(tissue)
    
    sol[tissue] = res['OptimizedValue']

plt.tight_layout()
plt.show()

# solve parameters
sol = pd.DataFrame( sol )
sol.index = ['sigma2/beta', 'beta']
sol.loc['sigma'] = np.sqrt( sol.loc['sigma2/beta'] * sol.loc['beta'] )
print( sol )
# -


