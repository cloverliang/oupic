from typing import Any, Callable, Dict

import numpy as np
import scipy.stats
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize


class LCEEstimator_array:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.alpha: float = config["alpha"]
        self.beta: float = config["beta"]
        self.omega: np.ndarray = config["omega"]
        self.nspecies: int = config["nspecies"]
        self.ncontrasts: int = self.nspecies - 1
        # TODO sanity check for number of species and length of variance vector
        # if(self.nspecies != (len(self.omega) + 1)):

    def fit_map_estimator(self, ya: np.ndarray, yb: np.ndarray) -> float:
        """
        Get MLE estimator, with contrast observations ya and yb in tissues a and b
        """

        def minus_log_likelihood(
            rho: float,
        ):  # a function that takes rho and returns a minus likelihood
            # return -self.get_log_likelihood(rho, ya, yb)
            return -self.get_log_likelihood_hardcode(rho, ya, yb)

        res = minimize(
            minus_log_likelihood, 0.0, method="L-BFGS-B", bounds=[(-0.99, 0.99)]
        )

        return res.x.item()

    def fit_mean_estimator(self, ya: np.ndarray, yb: np.ndarray) -> float:
        """
        Get posterior mean estimator, with contrast pbservations ya and yb in tissues a and b
        """
        return self._expectation_calc(self.get_posterior_pdf(ya, yb))

    def get_prior_mean(self) -> float:
        """
        Get prior mean using theoretical formula
        (2 * alpha / (alpha + beta) - 1)
        """
        return 2 * self.alpha / (self.alpha + self.beta) - 1

    def get_prior_pdf(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Get P(rho) as a function (Callable)
        """
        return self._prior_beta_density

    def get_posterior_pdf(
        self, ya: np.ndarray, yb: np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Get P(rho|ya,yb) as a function (Callable)
        """
        map_rho = self.fit_map_estimator(ya, yb)

        def func(rho):
            # post_pdf = self._normal_density(rho, ya,
            #                                 yb) * self._prior_beta_density(rho)
            post_pdf = np.exp(
                self.get_log_likelihood_hardcode(rho, ya, yb)
                - self.get_log_likelihood_hardcode(map_rho, ya, yb)
            )
            return post_pdf

        def func_norm(rho):
            return func(rho) / self._integration_calc(func)

        return func_norm

    def get_log_likelihood(
        self, rho: float, ya: np.ndarray, yb: np.ndarray
    ) -> np.ndarray:
        """
        DO NOT CALL, may cause numerical problems. Use get_log_likelihood_hardcode instead.
        Get log likelihood (not normalized) using internal functions
        """
        return np.log(self._normal_density(rho, ya, yb)) + np.log(
            self._prior_beta_density(rho)
        )

    def get_log_likelihood_hardcode(
        self, rho: float, ya: np.ndarray, yb: np.ndarray
    ) -> np.ndarray:
        """
        Get log likelihood by hard code, constant omited
        """
        ya_ = ya / np.sqrt(self.omega)
        yb_ = yb / np.sqrt(self.omega)

        normal_part = (
            -0.5 / (1 - rho ** 2) * (ya_ @ ya_ + yb_ @ yb_ - 2 * rho * (ya_ @ yb_))
        )
        beta_and_normal_part = (self.alpha - 1 - self.ncontrasts / 2) * np.log(
            1 + rho
        ) + (self.beta - 1 - self.ncontrasts / 2) * np.log(1 - rho)
        res_constant = (
            -0.5 * self.ncontrasts * np.log(2 * np.pi)
            - np.log(np.prod(self.omega))
            - np.log(2)
            - np.log(
                scipy.special.gamma(self.alpha)
                * scipy.special.gamma(self.beta)
                / scipy.special.gamma(self.alpha + self.beta)
            )
            - (self.alpha + self.beta - 2) * np.log(2)
        )

        return normal_part + beta_and_normal_part + res_constant

    def _normal_density(self, rho: float, ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        """
        Get P(y|rho)
        Args:
            rho (np.ndarray / float): correlation; could be passed as a vector
            ya (np.ndarray): gene expression contrast in tissue a, length of n-1
            yb (np.ndarray): gene expression contrast in tissue b, length of n-1
        """
        ya_ = ya / np.sqrt(self.omega)
        yb_ = yb / np.sqrt(self.omega)

        root_det = np.prod(self.omega) * ((1 - rho ** 2) ** (self.ncontrasts / 2))
        exp_A = ya_ @ ya_ + yb_ @ yb_
        exp_B = 2 * (ya_ @ yb_)

        normalized_fac = root_det * ((2 * np.pi) ** (self.ncontrasts / 2))

        return np.exp(-0.5 / (1 - rho ** 2) * (exp_A - exp_B * rho)) / normalized_fac

    def _normal_density_autoscale(
        self, rho: float, ya: np.ndarray, yb: np.ndarray
    ) -> np.ndarray:
        """
        Get P(y|rho)
        different with _normal_density, auto scaled to avoid numerical errors in calculation
        """

        ya_ = ya / np.sqrt(self.omega)
        yb_ = yb / np.sqrt(self.omega)

        root_det_autoscale = (1 - rho ** 2) ** (self.ncontrasts / 2)  # remove constant
        exp_A = ya_ @ ya_ + yb_ @ yb_
        exp_B = 2 * (ya_ @ yb_)
        exp_autoscale = np.exp(
            -0.5 / (1 - rho ** 2) * (exp_A - exp_B * rho) + 0.5 * exp_A
        )

        return exp_autoscale / root_det_autoscale

    def _prior_beta_density(self, rho: float) -> np.ndarray:
        """
        Get P(rho), beta distribution scaled to (-1, 1)
        Args:
            rho (np.ndarray): correlation; could be passed as an array
        """
        return 0.5 * scipy.stats.beta.pdf(0.5 * (rho + 1), self.alpha, self.beta)

    def _integration_calc(self, func: Callable[[np.ndarray], np.ndarray]) -> float:
        """
        Perform numerical integration for a single variable function between (-1, 1)
        """
        # TODO change to uniform integration
        xs, weight = leggauss(20)
        return func(xs) @ weight

    def _expectation_calc(self, func: Callable[[np.ndarray], np.ndarray]) -> float:
        """
        Perform numerical integration for a single variable function between (-1, 1)
        """
        # TODO change to uniform integration
        xs, weight = leggauss(20)
        return (func(xs) * xs) @ weight
