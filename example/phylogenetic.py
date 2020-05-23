# +
import dendropy as dp
import numpy as np
import pandas as pd

from typing import Dict, Callable
from pylce.pic import PIC
# -

tree = dp.Tree.get(path = '/Users/cong/Documents/Projects/2020-per-gene-LCE/data/brawand_tree.nwk', schema = 'newick')

# +
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
    
# check internal node iter
for nd in tree.postorder_internal_node_iter():
    print(nd.label, ', distance to parent: ', nd.edge_length)
    [a, b] = nd.child_nodes() 
    print(nd.num_child_nodes())
    print('Children:', a.label, ', ', b.label)
# -

tree.print_plot()

# +
# [c for c in dir(nd) if not c.startswith('_')] # check node content
# -

tree.gk_lambda = .2

test = {'Gallus gallus -chicken': 1,
'Ornithorhynchus anatinus -platypus': 2,
'Monodelphis domestica -opossum': 3,
'Mus musculus -mouse': 4,
'Pongo abelii -orangutan': 5,
'Gorilla gorilla -gorilla': 6,
'Homo sapiens -human': 7,
'Pan paniscus -bonobo': 8,
'Pan troglodytes -chimpanzee': 9,
'Macaca mulatta -macaque': 10}
test

pic = PIC(tree, test)
res = pic.calc_contrast()
np.exp(-10)

pd.DataFrame(res).T.sort_index()


