import dendropy as dp
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.spatial.distance import pdist, squareform


def phy_dist(x: float, A: float, B: float) -> float:
    return A * (1 - np.exp(-2 * B * x))


def fit_sigma_lambda_estimator(exp_data: pd.DataFrame, tree: dp.Tree):
    # species in expression data
    exp_species = exp_data.columns
    ngene = exp_data.shape[0]
    # species in tree
    tr_species = [sample.label for sample in tree.taxon_namespace]

    # euclidean distance
    exp_dist_df = (
        pd.DataFrame(
            squareform(pdist(exp_data.T, "sqeuclidean")),
            columns=exp_species,
            index=exp_species,
        )
        / ngene
    )

    # time distance
    pdc_dist = {}
    pdc = tree.phylogenetic_distance_matrix()
    for key0 in tree.taxon_namespace:
        tmp = {}
        for key1 in tree.taxon_namespace:
            tmp[key1.label] = pdc(key0, key1)
        pdc_dist[key0.label] = tmp
    pdc_df = pd.DataFrame.from_dict(pdc_dist)

    # rearrange & extract
    species = list(set(exp_species) & set(tr_species))
    exp_dist_df = exp_dist_df.loc[species, species]
    pdc_df = pdc_df.loc[species, species]

    # Extract training data
    xtrain = np.concatenate([row[i + 1 :] for i, row in enumerate(pdc_df.values)])
    ytrain = np.concatenate([row[i + 1 :] for i, row in enumerate(exp_dist_df.values)])

    res, res_hessian = optimize.curve_fit(phy_dist, xtrain, ytrain, (50, 0.001))

    return {
        "OptimizedValue": res,
        "hessian": res_hessian,
        "xtrain": xtrain,
        "ytrain": ytrain,
    }
