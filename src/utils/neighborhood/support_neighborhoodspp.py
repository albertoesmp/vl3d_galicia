# ---   IMPORTS   --- #
# ------------------- #
from src.utils.neighborhood.support_neighborhoods import SupportNeighborhoods
import pyvl3dpp as vl3dpp
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class SupportNeighborhoodsPP(SupportNeighborhoods):
    """
    :author: Alberto M. Esmoris Pena

    C++ version of :class:`.SupportNeighborhoods`.

    It supports more types of neighborhoods like bounded cylindrical, 2D-KNN,
    3D-KNN, bounded 2D-KNN, and bounded 3D-KNN neighborhoods.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, neighborhood_spec, **kwargs):
        # Call parent's init
        super().__init__(neighborhood_spec, **kwargs)

    # ---  NEIGHBORHOOD COMPUTATION METHODS  --- #
    # ------------------------------------------ #
    def compute(self, X, y=None, kdt=None):
        """
        C++ version of :meth:`.SupportNeighborhoods.compute`.
        """
        # Determine C++ function to be called
        Xdtype = X.dtype
        ydtype = np.int8 if y is None else y.dtype
        cpp_f = vl3dpp.alg_support_neighborhoods_fs32u32
        if Xdtype == np.float32:  # 32 bits structure space
            if ydtype == np.int8:  # 8 bits signed label
                cpp_f = vl3dpp.alg_support_neighborhoods_fs8u32
            elif ydtype == np.uint8:  # 8 bits unsigned label
                cpp_f = vl3dpp.alg_support_neighborhoods_fu8u32
            elif ydtype == np.int16:  # 16 bits signed label
                cpp_f = vl3dpp.alg_support_neighborhoods_fs16u32
            elif ydtype == np.uint16:  # 16 bits unsigned label
                cpp_f = vl3dpp.alg_support_neighborhoods_fu16u32
            elif ydtype == np.int32:  # 32 bits signed label
                cpp_f = vl3dpp.alg_support_neighborhoods_fs32u32
            else:  # 32 bits unsigned label
                cpp_f = vl3dpp.alg_support_neighborhoods_fu32u32
        else:  # 64 bits structure space
            if ydtype == np.int8:  # 8 bits signed label
                cpp_f = vl3dpp.alg_support_neighborhoods_ds8u32
            elif ydtype == np.uint8:  # 8 bits unsigned label
                cpp_f = vl3dpp.alg_support_neighborhoods_du8u32
            elif ydtype == np.int16:  # 16 bits signed label
                cpp_f = vl3dpp.alg_support_neighborhoods_ds16u32
            elif ydtype == np.uint16:  # 16 bits unsigned label
                cpp_f = vl3dpp.alg_support_neighborhoods_du16u32
            elif ydtype == np.int32:  # 32 bits signed label
                cpp_f = vl3dpp.alg_support_neighborhoods_ds32u32
            else:  # 32 bits unsigned label
                cpp_f = vl3dpp.alg_support_neighborhoods_du32u32
        # Prepare arguments for C++ call
        if y is None or not isinstance(y, np.ndarray):
            y = np.array(y)
        training_class_distribution = self.training_class_distribution
        if training_class_distribution is None:
            training_class_distribution = np.array([], dtype=np.int32)
        else:
            training_class_distribution = np.array(
                self.training_class_distribution
            )
        # Call C++
        sup_X, I = cpp_f(
            self.neighborhood_spec['type'],
            self.neighborhood_spec.get('k', 16),
            np.array(self.neighborhood_spec.get('radius', 1.0)),
            self.neighborhood_spec['separation_factor'],
            self.support_strategy,
            self.support_strategy_num_points,
            self.support_strategy_fast,
            training_class_distribution,
            self.center_on_pcloud,
            True,  # Extra nodes to match python-side support neighborhoods
            self.nthreads,
            X,
            y
        )
        # Return output
        return sup_X, I