from typing import Dict

import numpy as np

from .base_calculator import BaseCalculator, ContrastInfo


class OUcalculator(BaseCalculator):
    def __init__(self, ts_lambda: float):
        self.ts_lambda = ts_lambda

    def calc_contrast_raw(
        self, left_res: ContrastInfo, right_res: ContrastInfo
    ) -> Dict[str, float]:
        lvl = self.ts_lambda * left_res.dist_to_parent
        lvr = self.ts_lambda * right_res.dist_to_parent
        left_par = np.exp(-lvr)
        right_par = -np.exp(-lvl)
        contrast_value = left_par * left_res.nd_value + right_par * right_res.nd_value
        contrast = {
            "contrast_raw": contrast_value,
            "left_par": left_par,
            "right_par": right_par,
        }
        return contrast

    def calc_contrast_variance(
        self, left_res: ContrastInfo, right_res: ContrastInfo
    ) -> float:
        lvl = self.ts_lambda * left_res.dist_to_parent
        lvr = self.ts_lambda * right_res.dist_to_parent
        contrast_variance = (
            0.5 * np.exp(-2 * lvr) + 0.5 * np.exp(-2 * lvl) - np.exp(-2 * lvl - 2 * lvr)
        )
        return contrast_variance

    def calc_contrast_standardized(
        self, left_res: ContrastInfo, right_res: ContrastInfo
    ) -> Dict[str, float]:
        # repeat calculation to avoid numerical issue??
        lvl = self.ts_lambda * left_res.dist_to_parent
        lvr = self.ts_lambda * right_res.dist_to_parent
        contrast_sd = np.sqrt(self.calc_contrast_variance(left_res, right_res))
        left_par = np.exp(-lvr) / contrast_sd
        right_par = -np.exp(-lvl) / contrast_sd
        contrast_value = left_par * left_res.nd_value + right_par * right_res.nd_value
        contrast = {
            "contrast_standardized": contrast_value,
            "left_par": left_par,
            "right_par": right_par,
        }
        return contrast

    def calc_addition_dist_to_parent(
        self, left_res: ContrastInfo, right_res: ContrastInfo
    ) -> float:
        lvl = self.ts_lambda * left_res.dist_to_parent
        lvr = self.ts_lambda * right_res.dist_to_parent
        addition_dist_to_parent = (
            0.5
            / self.ts_lambda
            * (
                np.log(np.exp(2 * lvl + 2 * lvr) - 1)
                - np.log(np.exp(2 * lvl) + np.exp(2 * lvr) - 2)
            )
        )
        return addition_dist_to_parent

    def calc_nd_value(
        self, left_res: ContrastInfo, right_res: ContrastInfo
    ) -> Dict[str, float]:
        lvl = self.ts_lambda * left_res.dist_to_parent
        lvr = self.ts_lambda * right_res.dist_to_parent
        a = np.exp(lvr) - np.exp(-lvr)
        b = np.exp(lvl) - np.exp(-lvl)
        t = np.sqrt(
            (np.exp(2 * lvl) + np.exp(2 * lvr) - 2) * (1 - np.exp(-2 * lvl - 2 * lvr))
        )
        node_value = a / t * left_res.nd_value + b / t * right_res.nd_value
        node = {"node_value": node_value, "left_par": a / t, "right_par": b / t}
        return node
