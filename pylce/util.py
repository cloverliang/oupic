import numpy as np

__all__ = [
    'calc_contrast_value',
    'calc_contrast_variance',
    'calc_addition_dist_to_parent',
    'calc_nd_value',
]


def calc_contrast_value(left_res, right_res, gk_lambda):
    lvl = gk_lambda * left_res['dist_to_parent']
    lvr = gk_lambda * right_res['dist_to_parent']
    contrast_value = np.exp(-lvr) * left_res['nd_value'] - np.exp(
        -lvl) * right_res['nd_value']
    return contrast_value


def calc_contrast_variance(left_res, right_res, gk_lambda):
    lvl = gk_lambda * left_res['dist_to_parent']
    lvr = gk_lambda * right_res['dist_to_parent']
    contrast_variance = np.exp(-2 * lvr) + np.exp(
        -2 * lvl) - 2 * np.exp(-2 * lvl - 2 * lvr)
    return contrast_variance


def calc_addition_dist_to_parent(left_res, right_res, gk_lambda):
    lvl = gk_lambda * left_res['dist_to_parent']
    lvr = gk_lambda * right_res['dist_to_parent']
    addition_dist_to_parent = 0.5 / gk_lambda * (
        np.log(np.exp(2 * lvl + 2 * lvr) - 1) -
        np.log(np.exp(2 * lvl) + np.exp(2 * lvr) - 2))
    return addition_dist_to_parent


def calc_nd_value(left_res, right_res, gk_lambda):
    lvl = gk_lambda * left_res['dist_to_parent']
    lvr = gk_lambda * right_res['dist_to_parent']
    a = np.exp(lvr) + np.exp(-lvr)
    b = np.exp(lvl) + np.exp(-lvl)
    t = np.sqrt((np.exp(2 * lvl) + np.exp(2 * lvr) - 2) *
                (1 - np.exp(-2 * lvl - 2 * lvr)))
    node_value = a / t * left_res['nd_value'] + b / t * right_res['nd_value']
    return node_value