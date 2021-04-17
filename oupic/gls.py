import dendropy as dp
import numpy as np
from scipy import stats


# +
# Covariance matrix of BM tree
def BMcov(tree: dp.Tree) -> np.ndarray:
    pdc = dp.PhylogeneticDistanceMatrix(tree)
    pdc.compile_from_tree(tree)
    ndim = len(tree.taxon_namespace)
    Sigma = np.zeros([ndim, ndim])
    for i, t1 in enumerate(tree.taxon_namespace):
        for j, t2 in enumerate(tree.taxon_namespace):
            Sigma[i,j] = pdc(t1, t2)
    Sigma = (Sigma.max() - Sigma)/Sigma.max()
    return(Sigma)

# Covariance matrix of OU tree
def OUcov(tree: dp.Tree, 
          llambda: float) -> np.ndarray: 
    pdc = dp.PhylogeneticDistanceMatrix(tree)
    pdc.compile_from_tree(tree)
    ndim = len(tree.taxon_namespace)
    Sigma = np.zeros([ndim, ndim])
    for i, t1 in enumerate(tree.taxon_namespace):
        for j, t2 in enumerate(tree.taxon_namespace):
            Sigma[i,j] = np.exp( - llambda * pdc(t1, t2)) 
    return(Sigma)


# -

# fit gls model
def gls_fit(
    X: np.array, # row as features, column as samples 
    Y: np.array, # 1-d arrray, elements as different samples
    Sigma: np.array, # Covariance matrix for genearlized least squares
    add_intercept = False # Whether to add a row of ones or not to X
) -> list:
    if(add_intercept): # add intercept if needed
        X = np.vstack((np.ones(X.shape[X.ndim - 1]), X))
    
    # inverse of covariance matrix
    S_inv = np.linalg.inv(Sigma)
    
    if(X.ndim == 1): # when only one feature and no intercept
        # calculate beta hat
        M = 1 / (X @ S_inv @ np.transpose(X))
        bhat = M * X @ S_inv @ Y
        # calculate sd hat 
        yhat = X * bhat
        df = len(Y) - 1
        s2hat = np.transpose(Y - yhat) @ S_inv @ (Y - yhat) / df
        bcov = s2hat * M
        sdhat = np.sqrt(bcov)
    else:
        # calculate beta hat
        M = np.linalg.inv( X @ S_inv @ np.transpose(X) )
        bhat = M @ X @ S_inv @ Y
        # calculate sd hat
        yhat = np.transpose(X) @ bhat 
        df = len(Y) - np.linalg.matrix_rank(np.transpose(X) @ M @ X @ S_inv)
        s2hat = np.transpose(Y - yhat) @ S_inv @ (Y - yhat) / df
        bcov = s2hat * M
        sdhat = np.sqrt(np.diagonal(bcov))
    
    # calculate t value and p value
    tval = bhat / sdhat
    pval = 1 - np.abs(1 - 2*stats.t.cdf(tval, df))
    
    # variance explained
    # Warning for generalized models!!!
    RSS = (Y - yhat) @ (Y - yhat) / ((Y - Y.mean()) @ (Y - Y.mean()))
    
    return {'b-hat': bhat, 'sd-error': sdhat, 't-val': tval, 'p-val': pval, 'RSS': RSS}


