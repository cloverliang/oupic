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

# define covariance matrix for single trait
def calc_sigma_x_ou(tree, attr_x):
    sigma_x = attr_x['sigma_x']
    lambda_x = attr_x['lambda_x']

    nspecies = len(tree.taxon_namespace)
    sigma_mtx = pd.DataFrame( np.empty((nspecies, nspecies)) )
    sigma_mtx[:] = np.nan
    sigma_mtx.columns = [nd.label for nd in tree.taxon_namespace]
    sigma_mtx.index = [nd.label for nd in tree.taxon_namespace]

    pdm = tree.phylogenetic_distance_matrix()
    for taxon_x in tree.taxon_namespace:
        for taxon_y in tree.taxon_namespace:
            t_x = pdm(taxon_x, taxon_y)/2
            t_y = pdm(taxon_x, taxon_y)/2
            sigma_mtx.loc[taxon_x.label, taxon_y.label] = cov_ou(sigma_x, sigma_x, lambda_x, lambda_x, t_x, t_y, 1)
    return sigma_mtx

# define covariance matrix for two traits
def calc_sigma_xy_ou(tree, attr_xy):
    sigma_x = attr_xy['sigma_x']
    sigma_y = attr_xy['sigma_y']
    lambda_x = attr_xy['lambda_x']
    lambda_y = attr_xy['lambda_y']
    gamma_xy = attr_xy['gamma_xy']
    
    nspecies = len(tree.taxon_namespace)
    
    # init matrix sigma_xy = [ [A, D], [C, B] ]
    sigma_xy_A = np.zeros((nspecies,nspecies))
    sigma_xy_B = np.zeros((nspecies,nspecies))
    sigma_xy_C = np.zeros((nspecies,nspecies))
    sigma_xy_D = np.zeros((nspecies,nspecies))

    # get matrix
    pdm = tree.phylogenetic_distance_matrix()
    for i,taxon_a in enumerate(tree.taxon_namespace):
        for j,taxon_b in enumerate(tree.taxon_namespace):
            t_a = pdm(taxon_a, taxon_b)/2
            t_b = pdm(taxon_a, taxon_b)/2
            sigma_xy_A[i,j] = cov_ou(sigma_x, sigma_x, lambda_x, lambda_x, t_a, t_b, 1)
            sigma_xy_B[i,j] = cov_ou(sigma_y, sigma_y, lambda_y, lambda_y, t_a, t_b, 1)
            sigma_xy_C[i,j] = cov_ou(sigma_x, sigma_y, lambda_x, lambda_y, t_a, t_b, gamma_xy)
            sigma_xy_D[i,j] = cov_ou(sigma_y, sigma_x, lambda_y, lambda_x, t_a, t_b, gamma_xy)
    
    # merge to sigma_xy
    sigma_xy = pd.DataFrame( np.block([ [sigma_xy_A, sigma_xy_D], [sigma_xy_C, sigma_xy_B] ]) )
    sigma_xy.columns = [nd.label + '_x' for nd in tree.taxon_namespace] + [nd.label + '_y' for nd in tree.taxon_namespace]
    sigma_xy.index = [nd.label + '_x' for nd in tree.taxon_namespace] + [nd.label + '_y' for nd in tree.taxon_namespace]

    return sigma_xy

# calculate covariance matrix for BM model
def calc_sigma_xy_bm(tree, attr_xy):
    sigma_x = attr_xy['sigma_x']
    sigma_y = attr_xy['sigma_y']
    gamma_xy = attr_xy['gamma_xy']
    nspecies = len(tree.taxon_namespace)

    # init matrix sigma_xy = [ [A, D], [C, B] ]
    sigma_xy_A = np.zeros((nspecies,nspecies))
    sigma_xy_B = np.zeros((nspecies,nspecies))
    sigma_xy_C = np.zeros((nspecies,nspecies))
    sigma_xy_D = np.zeros((nspecies,nspecies))

    # get matrix
    pdm = tree.phylogenetic_distance_matrix()
    for i,nd_a in enumerate(tree.leaf_node_iter()):
        for j, nd_b in enumerate(tree.leaf_node_iter()):
            dist_to_root = .5 * ( nd_a.distance_from_root() + nd_b.distance_from_root() - pdm(nd_a.taxon, nd_b.taxon) )
            sigma_xy_A[i,j] = (sigma_x**2) * dist_to_root
            sigma_xy_B[i,j] = (sigma_y**2) * dist_to_root
            sigma_xy_C[i,j] = gamma_xy * sigma_x * sigma_y * dist_to_root
            sigma_xy_D[i,j] = gamma_xy * sigma_x * sigma_y * dist_to_root
    
    # merge to sigma_xy
    sigma_xy = pd.DataFrame( np.block([ [sigma_xy_A, sigma_xy_D], [sigma_xy_C, sigma_xy_B] ]) )
    sigma_xy.columns = [nd.label + '_x' for nd in tree.leaf_node_iter()] + [nd.label + '_y' for nd in tree.leaf_node_iter()]
    sigma_xy.index = [nd.label + '_x' for nd in tree.leaf_node_iter()] + [nd.label + '_y' for nd in tree.leaf_node_iter()]

    return sigma_xy

# define optima/expectation matrix
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
    # covariance matrix
    if attr_xy['mode'] == 'BM':
        sigma_xy = calc_sigma_xy_bm(tree, attr_xy)
    else:
        sigma_xy = calc_sigma_xy_ou(tree, attr_xy)
    # expectation matrix
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