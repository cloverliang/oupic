import abc
from dataclasses import dataclass
from typing import Dict

__all__ = ["ContrastInfo", "BaseCalculator"]


@dataclass
class ContrastInfo:
    is_contrast: bool
    contrast_standardized: float
    dist_to_parent: float
    nd_value: float


class BaseCalculator:
    @abc.abstractmethod
    def calc_contrast_raw(
        self, left_info: ContrastInfo, right_info: ContrastInfo
    ) -> Dict[str, float]:
        pass

    @abc.abstractmethod
    def calc_contrast_variance(
        self, left_info: ContrastInfo, right_info: ContrastInfo
    ) -> float:
        pass

    @abc.abstractmethod
    def calc_contrast_standardized(
        self, left_info: ContrastInfo, right_info: ContrastInfo
    ) -> Dict[str, float]:
        pass

    @abc.abstractmethod
    def calc_addition_dist_to_parent(
        self, left_info: ContrastInfo, right_info: ContrastInfo
    ) -> float:
        pass

    @abc.abstractmethod
    def calc_nd_value(
        self, left_info: ContrastInfo, right_info: ContrastInfo
    ) -> Dict[str, float]:
        pass
