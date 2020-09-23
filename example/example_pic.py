from pathlib import Path

import dendropy as dp

from oupic.ou_calculator import OUCalculator
from oupic.pic import PIC

# resolve file path
DIR_NAME = Path(__file__).resolve().parents[1]
filename = DIR_NAME / "data" / "brawand_tree.nwk"

# load tree
tree = dp.Tree.get_from_path(filename, schema="newick")

# generate sample data
taxa_val = {}
i = 1
for taxon in tree.taxon_namespace:
    taxa_val[taxon.label] = i
    i += 1

# initialize ou calculator with lambda=0.5
calculator = OUCalculator(ts_lambda=0.5)

# initialize pic
pic = PIC(tree, calculator, taxa_val)

pic.calc_contrast()


# print result
format_str = "=" * 30 + " %s " + "=" * 30

print(format_str % "node coef")
print(pic.node_coef)

print("\n" + format_str % "contrast coef")
print(pic.contrast_coef)

print("\n" + format_str % "contrast info")
print(pic.contrasts)
