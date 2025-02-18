# ---   IMPORTS   --- #
# ------------------- #
from src.mining.smooth_feats_miner import SmoothFeatsMiner
import pyvl3dpp as vl3dpp
from src.main.main_config import VL3DCFG
import src.main.main_logger as LOGGING
import numpy as np
import time


# ---   CLASS   --- #
# ----------------- #
class SmoothFeatsMinerPP(SmoothFeatsMiner):
    """
    :author: Alberto M. Esmoris Pena

    C++ version of the :class:`.SmoothFeatsMiner` data miner.

    It also supports more neighborhoods like 2D k-nearest neighbors, bounded
    cylindrical neighborhoods, and 2D and 3D rectangular neighborhoods.

    See :class:`.Miner` and :class:`.SmoothFeatsMiner`.
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_miner_args(spec):
        """
        Extract the arguments to initialize/instantiate a SmoothFeatsMinerPP
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a SmoothFeatsMinerPP.
        """
        # Return
        return SmoothFeatsMiner.extract_miner_args(spec)

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of SmoothFeatsMinerPP.

        See :meth:`.SmoothFeatsMiner.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)

    # ---   MINER METHODS   --- #
    # ------------------------- #
    def mine(self, pcloud):
        """
        Mine smooth features from the given point cloud.

        :param pcloud: The point cloud to be mined.
        :return: The point cloud extended with smooth features.
        :rtype: :class:`.PointCloud`
        """
        # Compute smooth features through C++ implementation
        start = time.perf_counter()
        # Obtain coordinates and features
        X = self.get_structure_space_matrix(pcloud)
        F = self.get_feature_space_matrix(pcloud, self.input_fnames)
        # Determine C++ method to be called
        structure_bits = VL3DCFG['MINING']['structure_space_bits']
        feature_bits = VL3DCFG['MINING']['feature_space_bits']
        cpp_f = vl3dpp.mine_smooth_feats_dd
        if structure_bits == 32:
            if feature_bits == 32:
                cpp_f = vl3dpp.mine_smooth_feats_ff
                LOGGING.LOGGER.debug(
                    'SmoothFeatsMinerPP will use a 32 bits structure space '
                    'and a 32 bits feature space.'
                )
            else:
                cpp_f = vl3dpp.mine_smooth_feats_fd
                LOGGING.LOGGER.debug(
                    'SmoothFeatsMinerPP will use a 32 bits structure space '
                    'and a 64 bits feature space.'
                )
        elif feature_bits == 32:
            cpp_f = vl3dpp.mine_smooth_feats_df
            LOGGING.LOGGER.debug(
                'SmoothFeatsMinerPP will use a 64 bits structure space '
                'and a 32 bits feature space.'
            )
        # Do the computation itself
        radius = self.neighborhood.get('radius', 0.25)
        Fhat = cpp_f(
            self.neighborhood.get('type', 'knn'),
            self.neighborhood.get('k', 16),
            np.array(
                radius if isinstance(radius, list) else
                [self.neighborhood.get('radius', 0.25)]*3
            ),
            self.neighborhood.get('lower_bound', -1.0),
            self.neighborhood.get('upper_bound', 1.0),
            self.weighted_mean_omega,
            self.gaussian_rbf_omega,
            self.nan_policy,
            self.fnames,
            self.nthreads,
            X,
            F
        )
        end = time.perf_counter()
        LOGGING.LOGGER.debug(
            f'SmoothFeatsMinerPP computed {Fhat.shape[1]} smoothed features '
            f'for {Fhat.shape[0]} points in {end-start:.3f} seconds.'
        )
        # Return point cloud extended with smooth features
        return pcloud.add_features(
            self.frenames,
            np.vstack(Fhat),
            ftypes=Fhat[0].dtype
        )
