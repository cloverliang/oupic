# +
import dendropy as dp
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from munch import Munch
from typing import Dict, Callable
from pylce.pic import PIC
# -

# tree = dp.Tree.get(path = '/Users/cong/Documents/Projects/2020-per-gene-LCE/data/brawand_tree.nwk', schema = 'newick')
# tree = dp.Tree.get(path = '/Users/cong/Documents/Projects/2020-per-gene-LCE/data/podos_tree.nwk', schema = 'newick')
tree = dp.Tree.get_from_path('/Users/cong/Documents/Projects/2020-per-gene-LCE/data/acer_species_modified_branch.nwk', schema = 'newick') # acer, L = 160, n = 55
# tree = dp.Tree.get_from_path('/Users/cong/Documents/Projects/2020-per-gene-LCE/data/pinus_species.nwk', schema = 'newick') # pinus, L = 73, n = 111

# +
# tree.print_plot()
# -

idx = 1
for nd in tree.postorder_node_iter():
    if(nd.taxon == None): ## Add internal label name
        nd.label = "Internal_" + str(idx)
        idx = idx + 1
    else:
        nd.label = nd.taxon.label
            
    try:
        # print(nd.taxon.label, nd.distance_from_root() - nd.parent_node.distance_from_root(), nd.edge_length)
        print(nd.taxon.label, nd.edge_length)
    except:
        print('+', nd.label,  nd.edge_length) # check iter function
    # break

test = {}
for species in tree.taxon_namespace:
    test[species.label] = 0

## vary the scale of values
L = 160
ts_lambda = np.array( [10/L, 3.3/L, 1/L, 0.33/L, 0.1/L] )
t_scale = ts_lambda * L
ts_lambda

# %matplotlib notebook

species = [sp.label for sp in tree.taxon_namespace]
nspecies = len(species)
xx, yy = np.meshgrid(np.arange(nspecies-1), np.arange(nspecies))

# +
attr = {'ts_lambda': np.nan}
res = {}
fig = plt.figure(figsize=(15, 6))
cmap = 'coolwarm'
alpha = 0.5
size_contrast = .1
size_node = 50

for i, ts_l in enumerate(ts_lambda):
    ax_contrast = fig.add_subplot(2, len(ts_lambda)+1, i + 1)
    ax_node = fig.add_subplot(2, len(ts_lambda)+1, i+2+len(ts_lambda))
    idx = 'LT=' + str(ts_l * L)
    
    attr['ts_lambda'] = ts_l
    picOU = PIC(tree, test, 'OU', attr)
    picOU.calc_contrast()
        
    # plot coef for contrast and node value
    zz_ou = np.abs(picOU.contrast_coef)*size_contrast /np.abs(picOU.contrast_coef.iloc[nspecies-1,nspecies-2])
    ou_color = picOU.contrast_coef.apply(lambda x: [0 if y <= 0 else 1 for y in x])
    ax_contrast.scatter(xx, yy, zz_ou, c = ou_color.values, cmap=cmap, alpha = alpha)
    ax_contrast.set_title(idx + ' contrast coef')
    
    zz_ou = np.abs(picOU.node_coef)*size_node/0.5
    ou_color = picOU.node_coef.apply(lambda x: [0 if y <= 0 else 1 for y in x])
    ax_node.scatter(xx, yy, zz_ou, c = ou_color.values, cmap = cmap, alpha = alpha)
    ax_node.set_title(idx + ' node coef')
    
attr = {}
picBM = PIC(tree, test, 'BM', attr)
picBM.calc_contrast()

ax_contrast = fig.add_subplot(2, len(ts_lambda)+1, len(ts_lambda) + 1)
zz_bm = np.abs(picBM.contrast_coef)*size_contrast / np.abs(picBM.contrast_coef.iloc[nspecies-1,nspecies-2])
bm_color = picBM.contrast_coef.apply(lambda x: [0 if y <= 0 else 1 for y in x])
ax_contrast.scatter(xx, yy, zz_bm, c = bm_color.values, cmap=cmap, alpha = alpha)
ax_contrast.set_title('BM contrast coef')
ax_node = fig.add_subplot(2, len(ts_lambda)+1, 2*len(ts_lambda)+2)

zz_bm = np.abs(picBM.node_coef)*size_node/0.5
bm_color = picBM.node_coef.apply(lambda x: [0 if y <= 0 else 1 for y in x])
ax_node.scatter(xx, yy, zz_bm, c = bm_color.values, cmap = cmap, alpha = alpha)
ax_node.set_title('BM node coef')

plt.tight_layout()
plt.show()
# -

