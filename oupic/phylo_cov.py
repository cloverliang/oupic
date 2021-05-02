import dendropy as dp
import numpy as np
import pandas as pd


class PhyloCov:
    def __init__(
        self,
        tree: dp.Tree
    ):
        '''
        Warning: 
        The tree must be ultrametric, i.e., all leaves are equidistant from the root.
        
        Example
        ------------------
        tree_cov = PhyloCov(tree)
        your_matrix = tree_cov.get_cov_mat(ts_lambda = 0.5)
        '''
        self.tree = tree
        self.species = [taxon.label for taxon in self.tree.taxon_namespace]
        self.pair_dist = self._get_pairwise_dist()
    
    def _get_pairwise_dist(self) -> pd.DataFrame:
        nspecies = len(self.species)
        pdm = dp.PhylogeneticDistanceMatrix(self.tree)
        pdm.compile_from_tree(self.tree)
        
        pair_dist = pd.DataFrame(np.zeros([nspecies, nspecies]),
                                index = self.species,
                                columns = self.species)
        for i, taxon_a in enumerate(self.tree.taxon_namespace):
            for j, taxon_b in enumerate(self.tree.taxon_namespace):
                pair_dist.iloc[i, j] = pdm(taxon_a, taxon_b)
        return(pair_dist)

    # covariance matrix of single trait
    def get_cov_mat(self, ts_lambda: float):
        '''
        Warning: Covariance matrices were normalized.
        '''
        if(ts_lambda == 0): # BM model
            tree_depth = self.pair_dist.max()
            Sigma = (tree_depth - self.pair_dist) / tree_depth
        elif(ts_lambda > 0): # OU model
            Sigma = np.exp( - ts_lambda * self.pair_dist)
        else:
            raise ValueError('lambda should be non-negative!')
        return(Sigma)


