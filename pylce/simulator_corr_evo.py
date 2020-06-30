import dendropy as dp
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from munch import Munch
from typing import Dict, Callable
from pylce.pic import PIC

# define covariance between two traits
def cov_ou(sigma_x, sigma_y, lambda_x, lambda_y, t_x, t_y, gamma_xy):
    return gamma_xy * sigma_x * sigma_y / (lambda_x + lambda_y) * np.exp(-lambda_x * t_x - lambda_y * t_y)

# define covariance matrix
def calc_sigma_xy(tree, attr_xy):
    sigma_x = attr_xy['sigma_x']
    sigma_y = attr_xy['sigma_y']
    lambda_x = attr_xy['lambda_x']
    lambda_y = attr_xy['lambda_y']
    gamma_xy = attr_xy['gamma_xy']
    
    nspecies = len(tree.taxon_namespace)
    
    # init sigma_xy
    sigma_xy = pd.DataFrame( np.empty((2*nspecies,2*nspecies)) )
    sigma_xy[:] = np.nan
    sigma_xy.columns = [nd.label + '_x' for nd in tree.taxon_namespace] + [nd.label + '_y' for nd in tree.taxon_namespace]
    sigma_xy.index = [nd.label + '_x' for nd in tree.taxon_namespace] + [nd.label + '_y' for nd in tree.taxon_namespace]
    
    # get matrix
    pdm = tree.phylogenetic_distance_matrix()
    for taxon_x in tree.taxon_namespace:
        for taxon_y in tree.taxon_namespace:
            t_x = pdm(taxon_x, taxon_y)/2
            t_y = pdm(taxon_x, taxon_y)/2
            sigma_xy.loc[taxon_x.label + '_x', taxon_y.label + '_x'] = cov_ou(sigma_x, sigma_x, lambda_x, lambda_x, t_x, t_y, 1)
            sigma_xy.loc[taxon_x.label + '_y', taxon_y.label + '_y'] = cov_ou(sigma_y, sigma_y, lambda_y, lambda_y, t_x, t_y, 1)
            sigma_xy.loc[taxon_x.label + '_x', taxon_y.label + '_y'] = cov_ou(sigma_x, sigma_y, lambda_x, lambda_y, t_x, t_y, gamma_xy)
            sigma_xy.loc[taxon_x.label + '_y', taxon_y.label + '_x'] = cov_ou(sigma_x, sigma_y, lambda_x, lambda_y, t_x, t_y, gamma_xy)
    return sigma_xy

# define optima matrix
def calc_mu_xy(tree, attr_xy):
    mu_x = attr_xy['mu_x']
    mu_y = attr_xy['mu_y']
    nspecies = len(tree.taxon_namespace)
    mu_xy = np.repeat([mu_x, mu_y], nspecies)
    return mu_xy

# define contrast coefficient matrix for ou model
def get_contrast_coef_ou(tree, attr):
    vec = {}
    for species in tree.taxon_namespace:
        vec[species.label] = 0
    pic_x = PIC(tree, vec, 'OU', attr)
    pic_x.calc_contrast()
    return pic_x.contrast_coef.T

# define contrast coefficient matrix for bm model
def get_contrast_coef_bm(tree):
    vec = {}
    for species in tree.taxon_namespace:
        vec[species.label] = 0
    pic_x = PIC(tree, vec, 'BM', attr={})
    pic_x.calc_contrast()
    return pic_x.contrast_coef.T

# define function for data simulation
def random_phylo_xy(tree, attr_xy, N):
    sigma_xy = calc_sigma_xy(tree, attr_xy)
    mu_xy = calc_mu_xy(tree, attr_xy)
    
    xy_rand = np.random.multivariate_normal(mu_xy, sigma_xy, N)
    xy_rand = pd.DataFrame(xy_rand)
    xy_rand.columns = sigma_xy.columns
    x_rand = xy_rand.filter(like = '_x')
    y_rand = xy_rand.filter(like = '_y')
    species_col = []
    for sample in x_rand.columns:
        species, tissue = sample.split('_x')
        species_col.append(species)
    x_rand.columns = species_col
    y_rand.columns = species_col
    return x_rand, y_rand, sigma_xy
    
def normalize_vec(mat):
    mean = np.mean(mat, axis=1)
    mat_ = mat.T - mean
    mat_ = mat_ / np.linalg.norm(mat_, axis=0)
    return mat_.T