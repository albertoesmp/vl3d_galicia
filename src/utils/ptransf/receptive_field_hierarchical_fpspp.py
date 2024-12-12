# ---   IMPORTS   --- #
# ------------------- #
from src.utils.ptransf.receptive_field_fpspp import ReceptiveFieldFPSPP
from src.utils.ptransf.receptive_field_fps import ReceptiveFieldFPS
from src.utils.ptransf.receptive_field_hierarchical_fps import \
    ReceptiveFieldHierarchicalFPS
from src.model.deeplearn.deep_learning_exception import DeepLearningException
import pyvl3dpp as vl3dpp
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class ReceptiveFieldHierarchicalFPSPP(ReceptiveFieldHierarchicalFPS):
    """
    :author: Alberto M. Esmoris Pena

    C++ implementation of the :class:`.ReceptiveFieldHierarchicalFPS`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        See :class:`.ReceptiveFieldHierarchicalFPS` and
        :meth:`.ReceptiveFieldHierarchicalFPS.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)

    # ---  RECEPTIVE FIELD METHODS  --- #
    # --------------------------------- #
    def fit(self, X, x, structure_float_type=np.float64, id=None):
        """
        C++ version of :meth:`.ReceptiveFieldHierarchicalFPS.fit`.
        """
        raise DeepLearningException(
            'ReceptiveFieldHierarchicalFPSPP fit method MUST not be called. '
            'The receptive fields are already fit when using as callable '
            'from C++.'
        )

    def propagate_values(self, v, reduce_strategy='mean', **kwargs):
        """
        See :class:`.ReceptiveFieldHierarchicalFPS` and
        :meth:`.ReceptiveFieldHierarchicalFPS.propagate_values`.
        """
        return ReceptiveFieldFPSPP.do_propagate_values(
            self.NUs[0], v, reduce_strategy
        )

    def reduce_values(self, X, v, reduce_f=np.mean):
        """
        C++ version of :meth:`.ReceptiveFieldHierarchicalFPS.reduce_values`.
        """
        return ReceptiveFieldFPSPP.do_reduce_values(
            self.NDs[0], X, v, reduce_f
        )

    def reduce_values_python(self, X, v, reduce_f=np.mean):
        """
        Method that calls
        :meth:`.ReceptiveFieldFPS.reduce_values` to provide a
        Python alternative to reductions.

        **NOTE** that this method should only be used for testing and debugging
        purposes.
        """
        return ReceptiveFieldFPS.do_reduce_values(
            self.NDs[0], X, v, reduce_f
        )
