import numpy as np
from scipy import stats


# fit gls model
def GLSFit(
    X: np.array, 
    Y: np.array, 
    Sigma_mat: np.array, 
    add_intercept = False, 
    Is_sigma_inverse = False
) -> list:
    '''
    Fit the GLS regression. Y ~ X * beta + epsilon.
    
    Warning: 
    1) Concordance between X, Y, Sigma_mat sample names were not checked.
    2) R-squared is controversal for GLS.
    
    Parameters:
    -------------
    X: np.array. row as features, column as samples.
    Y: np.array. Must be 1-d array, elements as different samples.
    Sigma_mat: np.array or pandas dataframe. Covariance matrix among samples.
    
    add_intercept: boolean. Whether to add a row of ones to X. Default = False.
    Is_sigma_inverse: boolean. When true, Sigma_mat is the inverse matrix 
            of the expected covariance matrix. Default = False
            
    Returns:
    ------------
    A dictionary containing:
    'b-hat': np.array. Estimated parameters.
    'sd-hat': np.array. Estimated standard deviation of parameters.
    't-val': np.array. t-value for significance test of b-hat = 0.
    'p-val': np.array. p-value for significance test of b-hat = 0.
    'R-squared': float. R-squared for the test.
    '''
    if(add_intercept): # add intercept if needed
        X = np.vstack((np.ones(X.shape[X.ndim - 1]), X))
    
    # inverse of covariance matrix
    if(Is_sigma_inverse):
        S_inv = Sigma_mat
    else:
        S_inv = np.linalg.inv(Sigma_mat)
    
    if(X.ndim == 1): # only one feature and no intercept
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
    Rsquared = (Y - yhat) @ S_inv @ (Y - yhat) / ((Y - Y.mean()) @ S_inv @ (Y - Y.mean()))
    
    sol = {'b-hat': bhat, 
           'sd-error': sdhat, 
           't-val': tval, 
           'p-val': pval, 
           'R-squared': Rsquared}
    return(sol)


