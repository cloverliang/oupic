from munch import Munch
from typing import Dict, Callable
import dendropy as dp
import numpy as np

# from .tree import Node, Tree
from .util import *


class PIC:
    def __init__(self, tree: dp.Tree, vec: Dict[str, float]):
        self._tree = tree
        self._vec = vec

    def calc_contrast(self):
        results = {}
        for nd in self._tree.postorder_node_iter():
            edge_length = nd.edge_length if nd.edge_length else 0

            # leaf
            hash_id = nd.label
            if nd.num_child_nodes() == 0:
                pars = {
                    'is_contrast': False,
                    'contrast_value': 0.,
                    'contrast_variance': 0.,
                    'normalized_contrast_value': 0.,
                    'dist_to_parent': edge_length, # modified distance to parent
                    'nd_value': self._vec[nd.label]
                }
                results[hash_id] = Munch(pars)
            # node with two child
            # contrast node
            else:
                # child_nodes() returns a list
                [left_child, right_child] = nd.child_nodes()
                left_res = results[left_child.label]
                right_res = results[right_child.label]

                contrast_value = calc_contrast_value(left_res, right_res, self._tree.gk_lambda)
                contrast_variance = calc_contrast_variance(left_res, right_res, self._tree.gk_lambda)
                nd_value = calc_nd_value(left_res, right_res, self._tree.gk_lambda)
                nd_addition_dist_to_parent = calc_addition_dist_to_parent(left_res, right_res, self._tree.gk_lambda)

                pars = {
                    'is_contrast': True,
                    'contrast_value': contrast_value,
                    'contrast_variance': contrast_variance,
                    'normalized_contrast_value': contrast_value / np.sqrt(contrast_variance),
                    'dist_to_parent': edge_length + nd_addition_dist_to_parent,
                    'nd_value': nd_value
                }
                results[hash_id] = Munch(pars)
        return results
