# ---   IMPORTS   --- #
# ------------------- #
from src.mining.miner import MinerException
from src.mining.recount_miner import RecountMiner
import pyvl3dpp as vl3dpp
from src.main.main_config import VL3DCFG
import src.main.main_logger as LOGGING
import numpy as np
import time


# ---   CLASS   --- #
# ----------------- #
class RecountMinerPP(RecountMiner):
    r"""
    :author: Alberto M. Esmoris Pena

    C++ version of the :class:`.RecountMiner` data miner.

    It supports more neighborhoods like 2D k-nearest neighbors, bounded
    cylindrical neighborhoods, and 2D and 3D rectangular neighborhoods.

    It also supports more recount-based features per filter like ring-based
    features, radial boundaries, 2D sectors, and 3D sectors.

    See :class:`.Miner` and :class:`.RecountMiner`.
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_miner_args(spec):
        """
        Extract the arguments to initialize/instantiate a RecountMinerPP
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a RecountMinerPP.
        """
        # Return
        return RecountMiner.extract_miner_args(spec)

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of RecountMinerPP.

        See :meth:`.RecountMiner.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)

    # ---   MINER METHODS   --- #
    # ------------------------- #
    def mine(self, pcloud):
        """
        Mine recount features from the given point cloud.

        :param pcloud: The point cloud to be mined.
        :return: The point cloud extended with recount-based features.
        :rtype: :class:`.PointCloud`
        """
        # Compute recount-based features through C++ implementation
        start = time.perf_counter()
        # Obtain coordinates and features
        X = self.get_structure_space_matrix(pcloud)
        F = self.get_feature_space_matrix(pcloud, self.input_fnames)
        # Determine C++ method to be called
        structure_bits = VL3DCFG['MINING']['structure_space_bits']
        feature_bits = VL3DCFG['MINING']['feature_space_bits']
        cpp_f = vl3dpp.mine_recount_dd
        if structure_bits == 32:
            if feature_bits == 32:
                cpp_f = vl3dpp.mine_recount_ff
                LOGGING.LOGGER.debug(
                    'RecountMinerPP will use a 32 bits structure space '
                    'and a 32 bits feature space.'
                )
            else:
                cpp_f = vl3dpp.mine_recount_fd
                LOGGING.LOGGER.debug(
                    'RecountMinerPP will use a 32 bits structure space '
                    'and a 64 bits feature space.'
                )
        elif feature_bits == 32:
            cpp_f = vl3dpp.mine_recount_df
            LOGGING.LOGGER.debug(
                'RecountMinerPP will use a 64 bits structure space '
                'and a 32 bits feature space.'
            )
        # Do the computation itself
        radius = self.neighborhood.get('radius', 0.25)
        cond_feat_indices, cond_types, cond_targets = \
            self.extract_cpp_conditions()
        Fhat = cpp_f(
            self.neighborhood.get('type', 'knn'),
            self.neighborhood.get('k', 16),
            np.array(
                radius if isinstance(radius, list) else
                [self.neighborhood.get('radius', 0.25)]*3
            ),
            self.neighborhood.get('lower_bound', -1.0),
            self.neighborhood.get('upper_bound', 1.0),
            [f.get('ignore_nan', False) for f in self.filters],
            [f.get('absolute_frequency', False) for f in self.filters],
            [f.get('relative_frequency', False) for f in self.filters],
            [f.get('surface_density', False) for f in self.filters],
            [f.get('volume_density', False) for f in self.filters],
            [f.get('vertical_segments', 0) for f in self.filters],
            [f.get('rings', 0) for f in self.filters],
            [f.get('radial_boundaries', 0) for f in self.filters],
            [f.get('sectors2D', 0) for f in self.filters],
            [f.get('sectors3D', 0) for f in self.filters],
            cond_feat_indices,
            cond_types,
            cond_targets,
            self.nthreads,
            X,
            F
        )
        # Return point cloud extended with recounts
        return pcloud.add_features(
            self.frenames, Fhat, ftypes=F.dtype
        )

    # ---   RECOUNT++ METHODS   --- #
    # ----------------------------- #
    def extract_cpp_conditions(self):
        """
        Obtain the conditions as arguments for the C++ recount miner.
        Each filter can have many conditions and each condition is represented
        by the index of the considered feature, the condition type, and the
        target value.

        See the :class:`.RecountMiner` documentation for more information.

        :return: Three different lists of lists. The first one for
            feature indices (integer), the second one for the condition types
            (string), and the third one for the target values (list of
            numbers). The first element of each list is a list whose elements
            define the different conditions for the corresponding filter.
        :rtype: tuple of three lists of lists
        """
        indices, types, targets = [], [], []
        for f in self.filters:
            _indices, _types, _targets = [], [], []
            if f['conditions'] is not None:
                for k, cond in enumerate(f['conditions']):
                    # Feature index
                    fname = cond['value_name']
                    idx = None
                    for i, fnamei in enumerate(self.input_fnames):
                        if fnamei == fname:
                            idx = i
                            break
                    if idx is None:
                        err_msg = 'RecountMinerPP received an unexpected '\
                            f'feature name "{fname}" for the condition {k}.'
                        LOGGING.LOGGER.error(err_msg)
                        raise MinerException(err_msg)
                    _indices.append(idx)
                    # Condition type
                    _types.append(cond['condition_type'])
                    # Target value
                    vt = cond['value_target']
                    if not isinstance(vt, list):
                        vt = [vt]
                    _targets.append(vt)
            indices.append(_indices)
            types.append(_types)
            targets.append(_targets)
        return indices, types, targets

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    def get_recount_names_from_filter(self, f):
        """
        Override the :meth:`.RecountMiner.get_recount_names_from_filter` to
        support the extra features generated by the C++ version.

        See :class:`.RecountMiner` and
        :meth:`.RecountMiner.get_recount_names_from_filter`.
        """

        filter_name = f['filter_name']
        filter_names = super().get_recount_names_from_filter(f)
        if f.get('rings', 0) > 0:
            filter_names.append(f'{filter_name}_rin')
        if f.get('radial_boundaries', 0) > 0:
            filter_names.append(f'{filter_name}_rb')
        if f.get('sectors2D', 0) > 0:
            filter_names.append(f'{filter_name}_st2')
        if f.get('sectors3D', 0) > 0:
            filter_names.append(f'{filter_name}_st3')
        return filter_names
