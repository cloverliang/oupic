import numpy as np
from scipy import stats

def get_lambda_from_kappa(kappa, L):
    return kappa / L * np.log(2)

# return leaf species in postorder
def get_species_postorder(tree):
    leaf_order = []
    for nd in tree.postorder_internal_node_iter():
        for child in nd.child_nodes():
            if child.is_leaf():
                leaf_order.append(child.label)
    return(leaf_order)

def normalize_vec(mat):
    mean = np.mean(mat, axis=1)
    mat_ = mat.T - mean
    mat_ = mat_ / np.linalg.norm(mat_, axis=0)
    return mat_.T

# get correlation between rows of two dataframes
def calc_row_correlation(x_pic, y_pic):
    normalized_x_pic = normalize_vec(x_pic)
    normalized_y_pic = normalize_vec(y_pic)
    return np.einsum('ij,ij -> i', normalized_x_pic, normalized_y_pic)

# get cutoff for certain alpha value
def get_r_cutoff_contrast(nspecies, alpha):
    df = nspecies -3 # ncontrasts = nspecies-1; df = ncontrasts-2
    # two-sided
    ll_t = stats.t.ppf(alpha/2, df)
    ll_r = ll_t / np.sqrt(df + ll_t**2)
    up_r = -ll_r
    return ll_r, up_r

def get_r_cutoff_tips(nspecies, alpha):
    df = nspecies - 2 
    # two-sided
    ll_t = stats.t.ppf(alpha/2, df)
    ll_r = ll_t / np.sqrt(df + ll_t**2)
    up_r = -ll_r
    return ll_r, up_r