# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.dlrun.furthest_point_subsampling_pre_processor \
    import FurthestPointSubsamplingPreProcessor
from src.model.deeplearn.dlrun.receptive_field_pre_processor import \
    ReceptiveFieldPreProcessor
from src.utils.ptransf.receptive_field_fpspp import ReceptiveFieldFPSPP
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
class FurthestPointSubsamplingPreProcessorPP(
    FurthestPointSubsamplingPreProcessor
):
    """
    :author: Alberto M. Esmoris Pena

    C++ implementation of the :class:`.FurthestPointSubsamplingPreProcessor`.
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
        training_class_distribution = \
            FurthestPointSubsamplingPreProcessorPP \
            .prepare_training_class_distribution(
                self.training_class_distribution
            )
        radii = FurthestPointSubsamplingPreProcessorPP.prepare_radii(
            self.neighborhood_spec
        )
        oversamplingArgs = FurthestPointSubsamplingPreProcessorPP \
            .prepare_oversampling(
            self.receptive_field_oversampling,
            self.num_points
        )
        Xdtype = X.dtype
        if Xdtype == np.float32:  # 32 bits for input structure space
            if structure_space_bits == 32:  # 32 bits for output structure
                cpp_f = vl3dpp.rf_dl_fps_preproc_Xff_Ff_Iu32u32_ys32
            else:  # 64 bits for output structure
                cpp_f = vl3dpp.rf_dl_fps_preproc_Xfd_Ff_Iu32u32_ys32
        else:  # 64 bits for input structure space
            if structure_space_bits == 32:  # 32 bits for output structure
                cpp_f = vl3dpp.rf_dl_fps_preproc_Xdf_Ff_Iu32u32_ys32
            else:  # 64 bits for output structure
                cpp_f = vl3dpp.rf_dl_fps_preproc_Xdd_Ff_Iu32u32_ys32
        # Call C++ to generate the receptive fields
        out = cpp_f(
            X,
            F.astype(np.float32),
            y.astype(np.int32),
            self.num_classes,
            self.to_unit_sphere,
            self.num_points,
            self.num_encoding_neighbors,
            self.fast,
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
        xout, Xout, Fout, yout, Nout, Mout, Iout = out
        Xout, Fout = np.array(Xout), np.array(Fout)
        if Fout.shape[-1] == 0:
            Fout = None
        if yout.shape[-1] == 0:
            yout = None
        if Iout[0].shape[1] == 1:
            for i in range(len(Iout)):
                Iout[i] = Iout[i].flatten()
        self.last_call_neighborhoods = Iout
        self.last_call_receptive_fields = []
        for i in range(Xout.shape[0]):
            rf = ReceptiveFieldFPSPP(
                num_points=self.num_points,
                num_encoding_neighbors=self.num_encoding_neighbors,
                fast=self.fast,
                receptive_field_oversampling=self.receptive_field_oversampling
            )
            rf.N = Nout[i]
            rf.M = Mout[i]
            rf.x = xout[i]
            rf.Y = Xout[i]
            self.last_call_receptive_fields.append(rf)
        # Report time
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'The C++ furthest point subsampling pre processor generated '
            f'{Xout.shape[0]} receptive fields of {self.num_points} points '
            f'each from {X.shape[0]} points in {end-start:.3f} seconds.'
        )
        # Export support points if requested
        if inputs.get('plots_and_reports', True):
            sup_X = np.vstack([
                rfi.x for rfi in self.last_call_receptive_fields
            ])
            self.export_support_points(inputs, sup_X)
        # Return with labels
        if yout is not None:
            if Fout is not None:  # Structure, features, and labels
                return [Xout, Fout], yout
            else:
                return Xout, yout  # Structure and labels
        # Return without labels
        if Fout is not None:
            return [Xout, Fout]  # Structure and features
        return Xout  # Structure only

    # ---  UTIL METHODS  --- #
    # ---------------------- #
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
                y, I[0], self.last_call_receptive_fields[0].N
            )
        # Validate C++ function
        if cpp_f is None:
            raise DeepLearningException(
                'FurthestPointSubsamplingPreProcessorPP failed to reduce '
                'labels due to incompatible data types.'
            )
        # Compute and return the reduced labels by calling C++
        return cpp_f(
            I,
            [rfi.N for rfi in self.last_call_receptive_fields],
            y,
            self.num_classes,
            self.nthreads
        )

    def reduce_labels_python(self, X_rf, y, I=None):
        """
        Method that mimics a call to
        :meth:`.FurthestPointSubsamplingPreProcessor.reduce_labels` to provide
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
                'FurthestPointSubsamplingPreProcessorPP cannot reduce labels '
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

    @staticmethod
    def find_cpp_reduce_label_function(y, Ii, N):
        """
        Determine the C++ function that must be used to reduce the point-wise
        labels considering the data types of the input.

        :param y: The input point-wise labels.
        :type y: :class:`np.ndarray`
        :param Ii: The indices of the neighbors in the original point cloud for
            a given i-th receptive field.
        :type Ii: :class:`np.ndarray`
        :param N: The first downsampling neighborhood of a given i-th receptive
            field.
        :type N: :class:`np.ndarray`
        :return: The C++ function for label reduction.
        """
        ydtype = y.dtype
        Ndtype = N.dtype
        Idtype = Ii.dtype
        p, q, r = None, None, None
        if ydtype == np.int8:
            p = 's8'
        elif ydtype == np.int16:
            p = 's16'
        elif ydtype == np.int32:
            p = 's32'
        elif ydtype == np.int64:
            p = 's64'
        elif ydtype == np.uint8:
            p = 'u8'
        elif ydtype == np.uint16:
            p = 'u16'
        elif ydtype == np.uint32:
            p = 'u32'
        elif ydtype == np.uint64:
            p = 'u64'
        if Idtype == np.uint8:
            q = 'u8'
        elif Idtype == np.uint16:
            q = 'u16'
        elif Idtype == np.uint32:
            q = 'u32'
        elif Idtype == np.uint64:
            q = 'u64'
        if Ndtype == np.uint8:
            r = 'u8'
        elif Ndtype == np.uint16:
            r = 'u16'
        elif Ndtype == np.uint32:
            r = 'u32'
        elif Ndtype == np.uint64:
            r = 'u64'
        return getattr(vl3dpp, f'rf_reduce_label_mode_{p}{q}{r}', None)

    @staticmethod
    def prepare_training_class_distribution(training_class_distribution):
        """
        Prepare the training class distribution to be used for a C++ deep
        learning pre-processing call.

        :param training_class_distribution: The training class distribution
            that must be prepared (typically, it comes from
            self.training_class_distribution)
        :return: Prepared training classs distribution.
        """
        if training_class_distribution is None:
            training_class_distribution = np.array([], dtype=np.int32)
        else:
            training_class_distribution = np.array(training_class_distribution)
        return training_class_distribution

    @staticmethod
    def prepare_radii(neighborhood_spec):
        """
        Prepare the radii argument to be used for a C++ deep learning
        pre-processing call.

        :param neighborhood_spec: The neighborhood specification that contains
            the information that is needed to prepare the radii argument.
        :type neighborhood_spec: dict
        :return: Prepared radii argument.
        :rtype: :class:`np.ndarray`
        """
        if (
            neighborhood_spec['type'].lower() == 'rectangular3d' or
            neighborhood_spec['type'].lower() == 'rectangular2d'
        ):
            radii = np.array([
                neighborhood_spec['radius'] for i in range(3)
            ])
        elif neighborhood_spec['type'].lower() == 'bounded_cylinder':
            radii = np.array([
                neighborhood_spec['radius'] for i in range(3)
            ])
            radii[1] *= -1
        else:
            radii = np.array([neighborhood_spec['radius']])
        return radii

    @staticmethod
    def prepare_oversampling(oversampling_spec, num_points):
        """
        Prepare the oversampling argument to be used for a C++ deep learning
        pre-processing call.

        :param oversampling_spec: The oversampling specification that contains
            the data that is needed to prepare the oversampling arguments.
        :type oversampling_spec: dict
        :param num_points: How many points are requested for the FPS subsampling
            strategy.
        :type num_points: int
        :return: Prepared oversampling arguments.
        :rtype: list
        """
        oversamplingArgs = []
        if oversampling_spec is not None:
            oversamplingArgs.append(
                oversampling_spec.get('min_points', 0)
            )
            oversamplingArgs.append(num_points)
            oversamplingArgs.append(
                oversampling_spec.get('strategy', 'nearest')
            )
            oversamplingArgs.append(
                oversampling_spec.get('k', 16)
            )
            oversamplingArgs.append(
                oversampling_spec.get('radius', 1.0)
            )
            oversamplingArgs.append(
                oversampling_spec.get('nthreads', 1)
            )
        return oversamplingArgs