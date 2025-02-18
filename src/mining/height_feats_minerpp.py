# ---   IMPORTS   --- #
# ------------------- #
from src.mining.height_feats_miner import HeightFeatsMiner
import pyvl3dpp as vl3dpp
from src.main.main_config import VL3DCFG
import src.main.main_logger as LOGGING
import time


# ---   CLASS   --- #
# ----------------- #
class HeightFeatsMinerPP(HeightFeatsMiner):
    """
    :author: Alberto M. Esmoris Pena

    C++ version of the :class:`.HeightFeatsMiner` data miner.

    See :class:`.Miner`.
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_miner_args(spec):
        """
        Extract the arguments to initialize/instantiate a HeightFeatsMinerPP
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a HeightFeatsMinerPP.
        """
        # Return
        return HeightFeatsMiner.extract_miner_args(spec)

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of HeightFeatsMinerPP.

        See :meth:`.HeightFeatsMiner.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)

    # ---  HEIGHT FEATURES METHODS  --- #
    # --------------------------------- #
    def compute_height_features(self, X):
        r"""
        Compute the height features using the C++ implementation.

        See :meth:`.HeightFeatsMiner.compute_height_features`.
        """
        # Compute height features through C++ implementation
        start = time.perf_counter()
        # Determine C++ method to be called
        structure_bits = VL3DCFG['MINING']['structure_space_bits']
        feature_bits = VL3DCFG['MINING']['feature_space_bits']
        cpp_f = vl3dpp.mine_height_feats_dd
        if structure_bits == 32:
            if feature_bits == 32:
                cpp_f = vl3dpp.mine_height_feats_ff
                LOGGING.LOGGER.debug(
                    'HeightFeatsMinerPP will use a 32 bits structure space '
                    'and a 32 bits feature space.'
                )
            else:
                cpp_f = vl3dpp.mine_height_feats_fd
                LOGGING.LOGGER.debug(
                    'HeightFeatsMinerPP will use a 32 bits structure space '
                    'and a 64 bits feature space.'
                )
        elif feature_bits == 32:
            cpp_f = vl3dpp.mine_height_feats_df
            LOGGING.LOGGER.debug(
                'HeightFeatsMinerPP will use a 64 bits structure space '
                'and a 32 bits feature space.'
            )
        # Do the computation itself
        F = cpp_f(
            self.neighborhood['type'].lower(),
            self.neighborhood['radius'],
            self.neighborhood['separation_factor'],
            '' if self.outlier_filter is None else self.outlier_filter,
            self.fnames,
            self.nthreads,
            X
        )
        end = time.perf_counter()
        LOGGING.LOGGER.debug(
            f'HeightFeatsMinerPP computed {F.shape[1]} height features for '
            f'{F.shape[0]} points in {end-start:.3f} seconds.'
        )
        # Return point-wise height features
        return F
