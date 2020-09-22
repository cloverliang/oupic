from typing import Dict, List

import dendropy as dp
import numpy as np
import pandas as pd

from .base_calculator import BaseCalculator, ContrastInfo, ValueWithCoef


class PIC:
    def __init__(
        self,
        tree: dp.Tree,
        calculator: BaseCalculator,
        taxa_val: Dict[str, float],
    ):
        self.tree: dp.Tree = tree
        self.taxa_val: Dict[str, float] = taxa_val
        self.calculator: BaseCalculator = calculator

        self.contrasts: Dict[str, ContrastInfo] = {}
        self.contrast_coef = None
        self.node_coef = None

    def calc_contrast(self):
        """
        Warning: 1) internal node name will be overwritten.
        2) only bifurcation trees allowed.
        """
        # tree: add internal node label and order leaves/species
        species = self._label_internal_nodes()

        # Initialize results
        contrasts: Dict[str, ContrastInfo] = {}
        contrast_coef: Dict[str, np.ndarray] = {}
        node_coef: Dict[str, np.ndarray] = {}
        for sp, row in zip(species, np.eye(len(species))):
            node_coef[sp] = row
            contrast_coef[sp] = row

        # postorder traversal nodes
        for nd in self.tree.postorder_node_iter():
            edge_length = nd.edge_length if nd.edge_length else 0

            # leaf
            if nd.num_child_nodes() == 0:
                contrasts[nd.label] = ContrastInfo(
                    is_contrast=False,
                    contrast_standardized=0.0,
                    dist_to_parent=edge_length,
                    nd_value=self.taxa_val[nd.label],
                )
            # internal node
            else:
                # child_nodes() returns a list
                # only bifurcation trees allowed
                left_child, right_child = nd.child_nodes()
                left_res = contrasts[left_child.label]
                right_res = contrasts[right_child.label]

                contrast_val: ValueWithCoef = self.calculator.calc_contrast(
                    left_res, right_res, standardized=True
                )
                nd_val: ValueWithCoef = self.calculator.calc_nd_value(
                    left_res, right_res
                )

                nd_addition_dist_to_parent = (
                    self.calculator.calc_addition_dist_to_parent(left_res, right_res)
                )

                contrasts[nd.label] = ContrastInfo(
                    is_contrast=True,
                    contrast_standardized=contrast_val.value,
                    dist_to_parent=edge_length + nd_addition_dist_to_parent,
                    nd_value=nd_val.value,
                )

                # add contrast and node value calculation parameters
                contrast_coef[nd.label] = (
                    contrast_val.left_par * node_coef[left_child.label]
                    + contrast_val.right_par * node_coef[right_child.label]
                )
                node_coef[nd.label] = (
                    nd_val.left_par * node_coef[left_child.label]
                    + nd_val.right_par * node_coef[right_child.label]
                )

        # reformat coefficient matrix to data frame
        self.contrast_coef = pd.DataFrame(contrast_coef, index=species).filter(
            like="Internal_"
        )
        self.node_coef = pd.DataFrame(node_coef, index=species).filter(like="Internal_")

        self.contrasts = pd.DataFrame(
            {label: contrast.to_dict() for label, contrast in contrasts.items()}
        ).filter(like="Internal_")

    def _label_internal_nodes(self) -> List[str]:
        idx = 1
        species = []
        for nd in self.tree.postorder_node_iter():
            if nd.taxon is None:
                # Add internal label
                nd.label = "Internal_" + str(idx)
                idx = idx + 1
                # identify leaves
                for child in nd.child_nodes():
                    if child.is_leaf():
                        species.append(child.label)
            else:
                # Add taxon label
                nd.label = nd.taxon.label
        return species
