# ---   IMPORTS   --- #
# ------------------- #
import numpy as np

from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.utils.ptransf.receptive_field_fps import ReceptiveFieldFPS
import pyvl3dpp as vl3dpp


# ---   CLASS   --- #
# ----------------- #
class ReceptiveFieldFPSPP(ReceptiveFieldFPS):
    """
    :author: Alberto M. Esmoris Pena

    C++ implementation of the :class:`.ReceptiveFieldFPS`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        See :class:`.ReceptiveFieldFPS` and
        :meth:`.ReceptiveFieldFPS.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)

    # ---   RECEPTIVE FIELD METHODS   --- #
    # ----------------------------------- #
    def fit(self, X, x, structure_float_type=np.float64, id=None):
        """
        C++ version of :meth:`.ReceptiveFieldFPS.fit`.
        """
        raise DeepLearningException(
            'ReceptiveFieldFPSPP fit method MUST not be called. '
            'The receptive fields are already fit when using as callable '
            'from C++.'
        )

    def propagate_values(self, v, reduce_strategy='mean', **kwargs):
        """
        C++ version of :meth:`.ReceptiveFieldFPS.propagate_values`.
        """
        return ReceptiveFieldFPSPP.do_propagate_values(
            self.M, v, reduce_strategy
        )

    @staticmethod
    def do_propagate_values(M, v, reduce_strategy):
        """
        See :meth:`.ReceptiveFieldFPSPP.propagate_values`.
        """
        Mdtype, vdtype = M.dtype, v.dtype
        reduce_strategy_low = reduce_strategy.lower()
        if reduce_strategy_low == 'mean':
            if vdtype == np.float32:  # 32 bits per feature
                if Mdtype == np.uint8:  # 8 bits per index
                    cpp_f = vl3dpp.rf_propagate_mean_f8
                elif Mdtype == np.uint16:  # 16 bits per index
                    cpp_f = vl3dpp.rf_propagate_mean_f16
                elif Mdtype == np.uint32:  # 32 bits per index
                    cpp_f = vl3dpp.rf_propagate_mean_f32
                else:  # 64 bits per index
                    cpp_f = vl3dpp.rf_propagate_mean_f64
            else:  # 64 bits per feature
                if Mdtype == np.uint8:  # 8 bits per index
                    cpp_f = vl3dpp.rf_propagate_mean_d8
                elif Mdtype == np.uint16:  # 16 bits per index
                    cpp_f = vl3dpp.rf_propagate_mean_d16
                elif Mdtype == np.uint32:  # 32 bits per index
                    cpp_f = vl3dpp.rf_propagate_mean_d32
                else:  # 64 bits per index
                    cpp_f = vl3dpp.rf_propagate_mean_d64
        elif reduce_strategy_low == 'closest':
            if vdtype == np.float32:  # 32 bits per feature
                if Mdtype == np.uint8:  # 8 bits per index
                    cpp_f = vl3dpp.rf_propagate_closest_f8
                elif Mdtype == np.uint16:  # 16 bits per index
                    cpp_f = vl3dpp.rf_propagate_closest_f16
                elif Mdtype == np.uint32:  # 32 bits per index
                    cpp_f = vl3dpp.rf_propagate_closest_f32
                else:  # 64 bits per index
                    cpp_f = vl3dpp.rf_propagate_closest_f64
            else:  # 64 bits per feature
                if Mdtype == np.uint8:  # 8 bits per index
                    cpp_f = vl3dpp.rf_propagate_closest_d8
                elif Mdtype == np.uint16:  # 16 bits per index
                    cpp_f = vl3dpp.rf_propagate_closest_d16
                elif Mdtype == np.uint32:  # 32 bits per index
                    cpp_f = vl3dpp.rf_propagate_closest_d32
                else:  # 64 bits per index
                    cpp_f = vl3dpp.rf_propagate_closest_d64
        else:
            raise ValueError(
                'The FPS++ receptive field received an unexpected '
                f'reduce_strategy "{reduce_strategy}" when propagating values.'
            )
        return cpp_f(M, v)

    def reduce_values(self, X, v, reduce_f=np.mean):
        """
        C++ version of :meth:`.ReceptiveFieldFPS.reduce_values`.
        """
        return ReceptiveFieldFPSPP.do_reduce_values(self.N, X, v, reduce_f)

    def reduce_values_python(self, X, v, reduce_f=np.mean):
        """
        Method that calls
        :meth:`.ReceptiveFieldFPS.reduce_values` to provide a Python alternative
        to reductions.

        **NOTE** that this method should only be used for testing and debugging
        purposes.
        """
        return ReceptiveFieldFPS.do_reduce_values(self.N, X, v, reduce_f)

    @staticmethod
    def do_reduce_values(N, X, v, reduce_f):
        """
        See :meth:`.ReceptiveFieldFPSPP.reduce_values`.
        """
        # Accelerate mean through C++
        if reduce_f == np.mean:
            Ndtype, vdtype = N.dtype, v.dtype
            if vdtype == np.float32:  # 32 bits per feature
                if Ndtype == np.uint8:  # 8 bits per index
                    cpp_f = vl3dpp.rf_reduce_mean_f8
                elif Ndtype == np.uint16:  # 16 bits per index
                    cpp_f = vl3dpp.rf_reduce_mean_f16
                elif Ndtype == np.uint32:  # 32 bits per index
                    cpp_f = vl3dpp.rf_reduce_mean_f32
                else:  # 64 bits per index
                    cpp_f = vl3dpp.rf_reduce_mean_f64
            else:  # 64 bits per feature
                if Ndtype == np.uint8:  # 8 bits per index
                    cpp_f = vl3dpp.rf_reduce_mean_d8
                elif Ndtype == np.uint16:  # 16 bits per index
                    cpp_f = vl3dpp.rf_reduce_mean_d16
                elif Ndtype == np.uint32:  # 32 bits per index
                    cpp_f = vl3dpp.rf_reduce_mean_d32
                else:  # 64 bits per index
                    cpp_f = vl3dpp.rf_reduce_mean_d64
            return cpp_f(N, v)
        # Delegate on python implementation
        return ReceptiveFieldFPS.do_reduce_values(
            N, X, v, reduce_f
        )



