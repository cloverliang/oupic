from typing import Callable, Dict

import numpy as np
import scipy.stats
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize


class LCEEstimator:
    def __init__(self, config: Dict[str, float]) -> None:
        self.alpha: float = config["alpha"]
        self.beta: float = config["beta"]
        self.omega: float = config["omega"]

    def fit_mle_estimator(self, y0: float, y1: float) -> float:
        """
        Get MLE estimator, with observation y0, y1
        """

        def minus_log_likelihood(
            rho: float,
        ):  # a function that takes rho and returns a minus likelihood
            return -self.get_log_likelihood(rho, y0, y1)

        res = minimize(
            minus_log_likelihood, 0.0, method="L-BFGS-B", bounds=[(-0.99, 0.99)]
        )
        return res.x

    def fit_mean_estimator(self, y0: float, y1: float) -> float:
        """
        Get posterior mean estimator, with observation y0, y1
        """
        return self._expectation_calc(self.get_posterior_pdf(y0, y1))

    def get_prior_pdf(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Get P(rho) as a function (Callable)
        """
        return self._prior_beta_density

    def get_posterior_pdf(
        self, y0: float, y1: float
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Get P(rho|y0,y1) as a function (Callable)
        """

        def func(rho):
            post_pdf = self._normal_density(rho, y0, y1) * self._prior_beta_density(rho)
            return post_pdf

        def func_norm(rho):
            return func(rho) / self._integration_calc(func)

        return func_norm

    def get_log_likelihood(self, rho: float, y0: float, y1: float) -> np.ndarray:
        """
        Get log likelihood (not normalized) using internal functions
        """
        return np.log(self._normal_density(rho, y0, y1)) + np.log(
            self._prior_beta_density(rho)
        )

    def _normal_density(self, rho: float, y0: float, y1: float) -> np.ndarray:
        """
        Get P(y|rho)
        Args:
            rho (np.ndarray / float): correlation; could be passed as a vector
            y0 (float): gene expression contrast
            y1 (float): gene expression contrast
        """
        root_det = 2 * self.omega * np.sqrt(1 - rho ** 2)
        return np.exp(
            -0.5 * (y0 ** 2 + y1 ** 2 - 2 * y0 * y1 * rho) / root_det ** 2
        ) / (2 * np.pi * root_det)

    def _prior_beta_density(self, rho: float) -> np.ndarray:
        """
        Get P(rho), beta distribution scaled to (-1, 1)
        Args:
            rho (np.ndarray): correlation; could be passed as an array
        """
        return 0.5 * scipy.stats.beta.pdf(0.5 * (rho + 1), self.alpha, self.beta)

    def _log_likelihood_hardcode(self, rho: float, y0: float, y1: float) -> np.ndarray:
        """
        Get log likelihood (proportional) by hard code
        """
        return {
            (self.alpha - 3 / 2) * np.log(1 + rho)
            + (self.beta - 3 / 2) * np.log(1 - rho)
            - 1
            / 4
            / self.omega
            * (y0 ** 2 + y1 ** 2 - 2 * y0 * y1 * rho)
            / (1 - rho ** 2)
        }

    def _integration_calc(self, func: Callable[[np.ndarray], np.ndarray]) -> float:
        """
        Perform numerical integration for a single variable function between (-1, 1)
        """
        xs, weight = leggauss(20)
        return func(xs) @ weight

    def _expectation_calc(self, func: Callable[[np.ndarray], np.ndarray]) -> float:
        """
        Perform numerical integration for a single variable function between (-1, 1)
        """
        xs, weight = leggauss(20)
        return (func(xs) * xs) @ weight
