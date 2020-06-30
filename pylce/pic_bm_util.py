import numpy as np
from munch import Munch


class BMcalculator:
    def __init__(self, attr):
        pass

    def calc_contrast_raw(self, left_res, right_res):
        contrast_raw = left_res['nd_value'] - right_res['nd_value']
        contrast = {'contrast_raw':contrast_raw, 'left_par':1, 'right_par':-1}
        return Munch(contrast)

    def calc_contrast_variance(self, left_res, right_res):
        return left_res['dist_to_parent'] + right_res['dist_to_parent']

    def calc_contrast_standardized(self, left_res, right_res):
        contrast_sd = np.sqrt( self.calc_contrast_variance(left_res, right_res) )
        left_par = 1/contrast_sd
        right_par = -1/contrast_sd
        contrast_standardized = left_par * left_res['nd_value'] + right_par * right_res['nd_value'] 
        contrast = {'contrast_standardized':contrast_standardized, 'left_par':left_par, 'right_par':right_par}
        return Munch(contrast)

    def calc_addition_dist_to_parent(self, left_res, right_res):
        vl = left_res['dist_to_parent']
        vr = right_res['dist_to_parent']
        addition_dist_to_parent = vl * vr / (vl + vr)
        return addition_dist_to_parent

    def calc_nd_value(self, left_res, right_res):
        vl = left_res['dist_to_parent']
        vr = right_res['dist_to_parent']
        left_par = vr / (vl + vr)
        right_par = vl / (vl + vr)
        node_value = left_par * left_res['nd_value'] + right_par * right_res['nd_value']
        node = {'node_value':node_value, 'left_par':left_par, 'right_par':right_par}
        return Munch(node)
