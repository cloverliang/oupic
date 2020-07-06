from munch import Munch
from typing import Dict, Callable
import dendropy as dp
import numpy as np
import pandas as pd

# from .tree import Node, Tree
from .pic_ou_util import OUcalculator 
from .pic_bm_util import BMcalculator


class PIC:
    def __init__(self, tree: dp.Tree, taxa_val: Dict[str, float], mode: str, attr: Dict[str, float]):
        self.tree = tree
        self.taxa_val = taxa_val
        self.mode = mode
        if mode == 'OU':
            calc_class = OUcalculator
        elif mode =='BM':
            calc_class = BMcalculator
        else:
            raise ValueError('Unkonwn mode %r.' % mode )
        self._calculator = calc_class(attr)

        self.contrasts = {}
        self.contrast_coef = None
        self.node_coef = None

    def calc_contrast(self): 
        '''
        Warning: 1) internal node name will be overwritten. 
        2) only bifurcation trees allowed.
        '''
        # tree: add internal node label and order leaves/species
        idx = 1
        species = []
        for nd in self.tree.postorder_node_iter():
            if(nd.taxon == None): 
                ## Add internal label
                nd.label = "Internal_" + str(idx)
                idx = idx + 1
                # identify leaves
                for child in nd.child_nodes():
                    if child.is_leaf():
                        species.append(child.label)
            else:
                # Add taxon label
                nd.label = nd.taxon.label

        # Initialize results
        contrasts = {}
        nspecies = len(species)
        contrast_coef = {}
        node_coef = {}
        for i,sp in enumerate(species):
            identity_vec = np.zeros(nspecies)
            identity_vec[i] = 1
            contrast_coef[sp] = identity_vec
            node_coef[sp] = identity_vec

        # postorder traversal nodes
        for nd in self.tree.postorder_node_iter():
            edge_length = nd.edge_length if nd.edge_length else 0

            # leaf
            if nd.num_child_nodes() == 0:
                pars = {
                    'is_contrast': False,
                    'contrast_standardized': 0.,
                    'dist_to_parent': edge_length, # modified distance to parent
                    'nd_value': self.taxa_val[nd.label]
                }
                contrasts[nd.label] = Munch(pars)
            # internal node
            else:
                # child_nodes() returns a list
                # only bifurcation trees allowed
                left_child, right_child = nd.child_nodes()
                left_res = contrasts[left_child.label]
                right_res = contrasts[right_child.label]

                contrast_val = self._calculator.calc_contrast_standardized(left_res, right_res)
                nd_val = self._calculator.calc_nd_value(left_res, right_res)
                nd_addition_dist_to_parent = self._calculator.calc_addition_dist_to_parent(left_res, right_res)

                pars = {
                    'is_contrast': True,
                    'contrast_standardized': contrast_val.contrast_standardized,
                    'dist_to_parent': edge_length + nd_addition_dist_to_parent,
                    'nd_value': nd_val.node_value
                }
                contrasts[nd.label] = Munch(pars)

                # add contrast and node value calculation parameters
                contrast_coef[nd.label] = contrast_val.left_par * node_coef[left_child.label] + contrast_val.right_par * node_coef[right_child.label]
                node_coef[nd.label] = nd_val.left_par * node_coef[left_child.label] + nd_val.right_par * node_coef[right_child.label]
        
        # reformat coefficient matrix to data frame
        contrast_coef = pd.DataFrame(contrast_coef)
        contrast_coef.index = species
        node_coef = pd.DataFrame(node_coef)
        node_coef.index = species

        self.contrast_coef = contrast_coef.filter(like = 'Internal_')
        self.node_coef = node_coef.filter(like = 'Internal_')
        self.contrasts = pd.DataFrame(contrasts).filter(like = 'Internal_')