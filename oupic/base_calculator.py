import abc
from dataclasses import dataclass

__all__ = ["ContrastInfo", "BaseCalculator"]


@dataclass
class ContrastInfo:
    __slots__ = ["is_contrast", "contrast_standardized", "dist_to_parent", "nd_value"]

    is_contrast: bool
    contrast_standardized: float
    dist_to_parent: float
    nd_value: float

    def to_dict(self):
        return {attr: self.__getattribute__(attr) for attr in type(self).__slots__}


@dataclass
class ValueWithCoef:
    __slots__ = ["value", "left_par", "right_par"]

    value: float
    left_par: float
    right_par: float

    def to_dict(self):
        return {attr: self.__getattribute__(attr) for attr in type(self).__slots__}


class BaseCalculator:
    @abc.abstractmethod
    def calc_contrast_variance(
        self, left_info: ContrastInfo, right_info: ContrastInfo
    ) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def calc_contrast(
        self,
        left_info: ContrastInfo,
        right_info: ContrastInfo,
        standardized: bool = True,
    ) -> ValueWithCoef:
        raise NotImplementedError

    @abc.abstractmethod
    def calc_addition_dist_to_parent(
        self, left_info: ContrastInfo, right_info: ContrastInfo
    ) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def calc_nd_value(
        self, left_info: ContrastInfo, right_info: ContrastInfo
    ) -> ValueWithCoef:
        raise NotImplementedError
