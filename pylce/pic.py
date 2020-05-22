from munch import Munch

from .tree import Node, Tree
from .util import *


class PIC:
    def __self__(self, tree: Tree, vec: Dict[str, float]):
        self._tree = tree
        self._vec = vec

    def calc_contrast(self):
        results = {}
        for nd in self._tree.post_order_traverse():
            # leaf
            hash_id = nd.name
            if nd.num_child == 0:
                pars = {
                    'is_contrast': False,
                    'contrast_value': 0.,
                    'contrast_variance': 0.,
                    'dist_to_parent': nd.dist_to_parent,
                    'nd_value': self._vec[hash_id]
                }
                results[hash_id] = Munch(pars)
            # node with two child
            # contrast node
            else:
                left_res = results[nd.left_child.name]
                right_res = results[nd.right_child.name]

                # need implememnt
                contrast_value = calc_contrast_value(left_res, right_res)
                contrast_variance = calc_contrast_variance(left_res, right_res)
                nd_value = calc_nd_value(left_res, right_res)
                nd_dist_to_parent = calc_dist_to_parent(left_res, right_res)

                pars = {
                    'is_contrast': True,
                    'contrast_value': contrast_value,
                    'contrast_variance': contrast_variance,
                    'dist_to_parent': nd_dist_to_parent,
                    'nd_value': nd_value
                }
                results[hash_id] = Munch(pars)
        return results
