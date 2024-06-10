# ---   IMPORTS   --- #
# ------------------- #
from src.utils.preds.prediction_reducer import PredictionReducer
from src.utils.preds.mean_pred_reduce_strategy import MeanPredReduceStrategy
from src.utils.preds.argmax_pred_select_strategy \
    import ArgMaxPredSelectStrategy


# ---   CLASS   --- #
# ----------------- #
class PredictionReducerFactory:
    """
    :author: Alberto M. Esmoris Pena

    Factory to build instances of :class:`.PredictionReducer`.
    """

    # ---   MAKE METHODS   --- #
    # ------------------------ #
    @staticmethod
    def make_from_dict(spec):
        """
        Make a :class:`.PredictionReducer` from the given dict-like
        specification.

        :param spec: The specification on how to build the prediction reducer.
        :return: The built prediction reducer.
        :rtype: :class:`.PredictionReducer.
        """
        pass  # TODO Rethink
