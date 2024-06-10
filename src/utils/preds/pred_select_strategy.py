# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod


# ---   CLASS   --- #
# ----------------- #
class PredSelectStrategy:
    """
    :author: Alberto M. Esmoris Pena

    Interface for select operations on predictions.

    See :class:`.PredictionReducer`.
    """
    # ---   INIT   --- #
    # ⨪--------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a prediction selection strategy.

        :param kwargs: The attributes for the PredSelectStrategy.
        """
        pass

    # ---  SELECT METHODS  --- #
    # ------------------------ #
    @abstractmethod
    def select(self, reducer, Z):
        r"""
        The method that provides the logit to select the values of interest
        from the reduced predictions. It must be overridenn by any concrete
        implementation of a prediction selection strategy.

        :param reducer: The prediction reducer that is doing the reduction.
        :type reducer: :class:`.PredictionReducer`
        :param Z: Matrix-like array. There are as many rows as points and
            as many columns as reduced predictions.
        :return: The selected predictions derived from the reduced predictions
            as a matrix. Either a matrix with the :math:`n_y` point-wise output
            values :math:`\pmb{\hat{Y}} \in \mathbb{R}^{m \times n_y}` or a
            vector for the case of a single point-wise scalar output
            :math:`\pmb{\hat{y}} \in \mathbb{R}^{m}`.
        :rtype: :class:`np.ndarray`
        """
        pass
