import numpy as np

from .base_calculator import BaseCalculator, ContrastInfo, ValueWithCoef


class BMCalculator(BaseCalculator):
    def calc_contrast_variance(
        self, left_res: ContrastInfo, right_res: ContrastInfo
    ) -> float:
        return left_res.dist_to_parent + right_res.dist_to_parent

    def calc_contrast(
        self, left_res: ContrastInfo, right_res: ContrastInfo, standardized: bool = True
    ) -> ValueWithCoef:
        contrast_sd = 1.0
        if standardized:
            contrast_sd = np.sqrt(self.calc_contrast_variance(left_res, right_res))
        left_par = 1 / contrast_sd
        right_par = -1 / contrast_sd
        contrast_standardized = (
            left_par * left_res.nd_value + right_par * right_res.nd_value
        )
        return ValueWithCoef(contrast_standardized, left_par, right_par)

    def calc_addition_dist_to_parent(
        self, left_res: ContrastInfo, right_res: ContrastInfo
    ) -> float:
        vl = left_res.dist_to_parent
        vr = right_res.dist_to_parent
        addition_dist_to_parent = vl * vr / (vl + vr)
        return addition_dist_to_parent

    def calc_nd_value(
        self, left_res: ContrastInfo, right_res: ContrastInfo
    ) -> ValueWithCoef:
        vl = left_res.dist_to_parent
        vr = right_res.dist_to_parent
        left_par = vr / (vl + vr)
        right_par = vl / (vl + vr)
        node_value = left_par * left_res.nd_value + right_par * right_res.nd_value
        return ValueWithCoef(node_value, left_par, right_par)
