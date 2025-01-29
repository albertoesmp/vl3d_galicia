# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.dlrun.furthest_point_subsampling_pre_processorpp \
    import FurthestPointSubsamplingPreProcessorPP
from src.model.deeplearn.dlrun.hierarchical_fps_pre_processor import \
    HierarchicalFPSPreProcessor
from src.model.deeplearn.dlrun.receptive_field_pre_processor import \
    ReceptiveFieldPreProcessor
from src.utils.ptransf.receptive_field_hierarchical_fpspp import \
    ReceptiveFieldHierarchicalFPSPP
from src.model.deeplearn.dlrun.grid_subsampling_pre_processor import \
    GridSubsamplingPreProcessor
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.main.main_config import VL3DCFG
import src.main.main_logger as LOGGING
import pyvl3dpp as vl3dpp
import joblib
import scipy
import numpy as np
import time


# ---   CLASS   --- #
# ----------------- #
class HierarchicalFPSPreProcessorPP(HierarchicalFPSPreProcessor):
    """
    :author: Alberto M. Esmoris Pena

    C++ version of the :class:`.HierarchicalFPSPreProcessor`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        C++ version of :meth:`.FurthestPointSubsamplingPreProcessor.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)

    # ---   RUN/CALL   --- #
    # -------------------- #
    def __call__(self, inputs):
        """
        C++ version of :meth:`.FurthestPointSubsamplingPreProcessor.__call__`.
        """
        LOGGING.LOGGER.info(
            'Generating hierarchical FPS receptive fields using the C++ '
            'extensions ...'
        )
        # Extract inputs
        start = time.perf_counter()
        X, F, y = inputs['X'], None, inputs.get('y', None)
        if isinstance(X, list):
            X, F = X[0], X[1]
        # Determine number of classes if not available
        ReceptiveFieldPreProcessor.num_classes_from_pwise_labels(self, y)
        # Determine float type for the structure space
        structure_space_bits = \
            VL3DCFG['MODEL']['ReceptiveField']['structure_space_bits']
        # Purge old state before computing new one
        self.purge_receptive_fields()
        # Prepare C++ call
        F = np.array([]) if F is None else F
        y = np.array([]) if y is None else y
        training_class_distribution = FurthestPointSubsamplingPreProcessorPP\
            .prepare_training_class_distribution(
                self.training_class_distribution
            )
        radii = FurthestPointSubsamplingPreProcessorPP.prepare_radii(
            self.neighborhood_spec
        )
        oversamplingArgs = FurthestPointSubsamplingPreProcessorPP\
            .prepare_oversampling(
                self.receptive_field_oversampling,
                self.num_points_per_depth[0]
            )
        Xdtype = X.dtype
        if Xdtype == np.float32:  # 32 bits for input structure space
            if structure_space_bits == 32:  # 32 bits for output structure
                cpp_f = vl3dpp.rf_dl_hfps_preproc_Xff_Ff_Iu32u32_ys32
            else:  # 64 bits for output structure
                cpp_f = vl3dpp.rf_dl_hfps_preproc_Xfd_Ff_Iu32u32_ys32
        else:  # 64 bits for input structure space
            if structure_space_bits == 32:  # 32 bits for output structure
                cpp_f = vl3dpp.rf_dl_hfps_preproc_Xdf_Ff_Iu32u32_ys32
            else:  # 64 bits for output structure
                cpp_f = vl3dpp.rf_dl_hfps_preproc_Xdd_Ff_Iu32u32_ys32
        # Call C++ to generate the receptive fields
        out = cpp_f(
            X,
            F.astype(np.float32),
            y.astype(np.int32),
            self.num_classes,
            self.to_unit_sphere,
            np.array(self.num_points_per_depth),
            np.array(self.num_downsampling_neighbors),
            np.array(self.num_upsampling_neighbors),
            np.array(self.num_pwise_neighbors),
            self.fast_flag_per_depth,
            [  # Support args
                self.neighborhood_spec['type'].lower(),
                self.neighborhood_spec.get('K', 16),
                radii,
                self.neighborhood_spec['separation_factor'],
                self.support_strategy.lower(),
                self.support_strategy_num_points,
                self.support_strategy_fast,
                training_class_distribution,
                self.center_on_pcloud,
                getattr(self, 'support_extra_nodes', True)
            ],
            oversamplingArgs,
            self.nthreads
        )
        # Assign outputs
        xout, Fout, yout, Iout, Xout, NDout, NUout, Nout = out
        Xout, Fout = [np.array(Xouti) for Xouti in Xout], np.array(Fout)
        if Fout.shape[-1] == 0:
            Fout = None
        if yout.shape[-1] == 0:
            yout = None
        if Iout[0].shape[1] == 1:
            for i in range(len(Iout)):
                Iout[i] = Iout[i].flatten()
        NDout = [np.array(NDouti) for NDouti in NDout]
        NUout[1:] = [np.array(NUouti) for NUouti in NUout[1:]]
        Nout = [np.array(Nouti) for Nouti in Nout]
        # Optimize the indexing of the pre-processed batches too
        start_iopt = time.perf_counter()
        HierarchicalFPSPreProcessorPP.optimize_indices(
            self.depth, NDout, NUout, Nout
        )
        end_iopt = time.perf_counter()
        LOGGING.LOGGER.info(
            'HierarchicalFPSPreProcessorPP optimized the neighborhood indexing '
            f'in {end_iopt-start_iopt:.3f} seconds.'
        )
        # Store encoding of original neighborhoods and create receptive fields
        self.last_call_neighborhoods = Iout
        self.last_call_receptive_fields = []
        for i in range(Xout[0].shape[0]):
            # Create i-th receptive field
            rf = ReceptiveFieldHierarchicalFPSPP(
                num_points_per_depth=self.num_points_per_depth,
                num_downsampling_neighbors=self.num_downsampling_neighbors,
                num_pwise_neighbors=self.num_pwise_neighbors,
                num_upsampling_neighbors=self.num_upsampling_neighbors,
                fast_flag_per_depth=self.fast_flag_per_depth,
                receptive_field_oversampling=self.receptive_field_oversampling
            )
            rf.NDs = [NDout[d][i] for d in range(self.depth)]
            rf.NUs = [NUout[d][i] for d in range(self.depth)]
            rf.Ns = [Nout[d][i] for d in range(self.depth)]
            rf.x = xout[i]
            rf.Ys = [Xout[d][i] for d in range(self.depth)]
            # Track receptive field
            self.last_call_receptive_fields.append(rf)
        # Report time
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'The C++ hierarchical furthest point subsampling pre processor '
            f'generated {Xout[0].shape[0]} receptive fields of '
            f'{self.num_points_per_depth} '
            f'points each from {X.shape[0]} points in {end-start:.3f} seconds.'
        )
        # Export support points if requested
        if inputs.get('plots_and_reports', True):
            sup_X = np.vstack([
                rfi.x for rfi in self.last_call_receptive_fields
            ])
            self.export_support_points(inputs, sup_X)
        # Prepare return
        pyout = [Xout[0], Fout] + Xout[1:] + NDout[1:] + Nout + NUout[1:]
        # Return with labels
        if yout is not None:
            return pyout, yout
        # Return without labels
        return pyout

    # ---  UTIL METHODS  --- #
    # ---------------------- #
    @staticmethod
    def optimize_indexing_memory(I):
        """
        Optimize the encoding of the received array of indices (I) to use as
        few bytes as possible. This method assists
        :meth:`.HierarchicalFPSPreProcessorPP.optimize_indices`.

        :param I: The array of integer indices whose memory encoding must be
            optimized.
        :type I: :class:`np.ndarray` of int
        """
        Imax = np.max(I)
        if Imax < 256:
            return I.astype(np.uint8)
        elif Imax < 65536:
            return I.astype(np.uint16)
        elif Imax < 4294967296:
            return I.astype(np.uint32)
        return I

    @staticmethod
    def optimize_indices(depth, NDs, NUs, Ns):
        """
        Optimize the memory required to encode the given hierarchical
        neighborhoods. This method is assisted by
        :meth:`.HierarchicalFPSPreProcessorPP.optimize_indexing_memory`.

        :param depth: The depth of the hierarchy.
        :type depth: int
        :param NDs: List whose elements are the downsampling neighborhoods at
            each depth, i.e. NDs[d] gives the downsampling neighborhoods at
            depth d.
        :type NDs: list
        :param NUs: List whose elements are the upsampling neighborhoods at
            each depth, i.e. NUs[d] gives the upsampling neighborhoods at
            depth d.
        :type NUs: list
        :param Ns: List whose elements are the neighborhoods at a given depth,
            i.e. Ns[d] gives the upsampling neighborhoods at depth d.
        :type Ns: list
        :return: Nothing, but the hierarchical neighborhods (NDs, NUs, Ns) are
            updated in place.
        """
        for d in range(depth):
            NDs[d] = HierarchicalFPSPreProcessorPP.optimize_indexing_memory(
                NDs[d]
            )
            if d > 0:
                NUs[d] = HierarchicalFPSPreProcessorPP.optimize_indexing_memory(
                    NUs[d]
                )
            Ns[d] = HierarchicalFPSPreProcessorPP.optimize_indexing_memory(
                Ns[d]
            )

    def reduce_labels(self, X_rf, y, I=None):
        """
        C++ version of
        :meth:`.FurthestPointSubsamplingPreProcessor.reduce_labels`.
        """
        # Handle automatic neighborhoods from cache
        if I is None:
            I = self.last_call_neighborhoods
        # Validate neighborhoods are given
        if I is None or len(I) < 1:
            raise DeepLearningException(
                'FurthestPointSubsamplingPreProcessorPP cannot reduce labels '
                'because no neighborhood indices were given.'
            )
        # Determine C++ function
        cpp_f = FurthestPointSubsamplingPreProcessorPP\
            .find_cpp_reduce_label_function(
                y, I[0], self.last_call_receptive_fields[0].NDs[0]
            )
        # Validate C++ function
        if cpp_f is None:
            raise DeepLearningException(
                'HierarchicalFPSPreProcessorPP failed to reduce '
                'labels due to incompatible data types.'
            )
        # Compute and return the reduced labels by calling C++
        return cpp_f(
            I,
            [rfi.NDs[0] for rfi in self.last_call_receptive_fields],
            y,
            self.num_classes,
            self.nthreads
        )

    def reduce_labels_python(self, X_rf, y, I=None):
        """
        Method that mimics a call to
        :meth:`.HierarchicalFPSPreProcessor.reduce_labels` to provide
        a Python alternative to label reduction.

        **NOTE** that this method should only be used for testing and debugging
        purposes.
        """
        # Handle automatic neighborhoods from cache
        if I is None:
            I = self.last_call_neighborhoods
        # Validate neighborhoods are given
        if I is None or len(I) < 1:
            raise DeepLearningException(
                'HierarchicalFPSPreProcessorPP cannot reduce labels '
                'through PYthon because no neighborhood indices were given.'
            )
        # Compute and return the reduced labels
        return np.array(joblib.Parallel(n_jobs=self.nthreads)(
            joblib.delayed(
                self.last_call_receptive_fields[i].reduce_values_python
            )(
                X_rf[i],
                y[Ii],
                reduce_f=lambda x: scipy.stats.mode(x)[0]
            ) for i, Ii in enumerate(I)
        ))
