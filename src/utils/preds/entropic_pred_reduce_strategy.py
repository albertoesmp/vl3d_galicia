# ---   IMPORTS   --- #
# ------------------- #
from src.utils.preds.pred_reduce_strategy import PredReduceStrategy
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class EntropicPredReduceStrategy(PredReduceStrategy):
    r"""
    :author: Alberto M. Esmoris Pena

    Reduce many predictions per point to a single one by considering the
    point-wise entropies.

    The reduced prediction for the :math:`j`-th class of the :math:`i`-th point
    will be as shown below, assuming :math:`K` values for the reduction.

    TODO Rethink : Add doc from LaTeX here

    See :class:`.PredReduceStrategy`.
    """
    # TODO Rethink : Add doc from LaTeX
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a mean prediction reduction strategy.

        :param kwargs: The attributes for the EntropicPredReduceStrategy.
        """
        # Call parent's init
        super().__init__(**kwargs)

    # ---  REDUCE METHODS  --- #
    # ------------------------ #
    def reduce(self, reducer, npoints, nvals, Z, I):
        """
        See :class:`.PredReduceStrategy` and
        :meth:`.PredReduceStrategy.reduce`.
        """
        # Initialize final likelihoods
        Zhat = np.zeros((npoints, nvals), dtype=float) if nvals > 1 \
            else np.zeros(npoints, dtype=float)
        # Initialize final denominators
        denoms = np.zeros((npoints, nvals), dtype=float) if nvals > 1 \
            else np.zeros(npoints, dtype=float)
        # Compute entropic norm
        Enorm = -nvals * np.exp(-1)*np.log2(np.exp(-1))
        # For each neighborhood
        for i, Zi in enumerate(Z):
            # Compute normalized entropies
            E = - Zi * np.log2(Zi)/Enorm
            # Compute weights from normalized entropies
            w = 1 - E
            # Aggregate final likelihoods
            Zhat[I[i]] += w*E
            denoms[I[i]] += w
        # Normalize final likelihoods
        non_zero_mask = denoms != 0
        Zhat[non_zero_mask] = Zhat[non_zero_mask]/denoms[non_zero_mask]
        return Zhat
