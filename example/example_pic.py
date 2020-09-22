import sys
sys.path.append('../')

import dendropy as dp
import numpy as np

from pylce.pic import PIC
from pylce.ou_calculator import OUCalculator
from pylce.bm_calculator import BMCalculator


tree = dp.Tree.get_from_path('../data/brawand_tree.nwk', schema = 'newick')

taxa_val = {}
i = 1
for taxon in tree.taxon_namespace:
    taxa_val[taxon.label] = i
    i += 1

calculator = OUCalculator(ts_lambda=0.5)
pic = PIC(tree, calculator, taxa_val)
pic.calc_contrast()


format_str = "=" * 20 + " %s " + "=" * 20 

print(format_str % "node coef")
print(pic.node_coef)

print("\n" + format_str % "contrast coef")
print(pic.contrast_coef)

print("\n" + format_str % "contrast info")
print(pic.contrasts)
