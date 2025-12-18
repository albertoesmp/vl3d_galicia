# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.dlrun.furthest_point_subsampling_pre_processorpp \
    import FurthestPointSubsamplingPreProcessorPP
from src.model.deeplearn.dlrun.receptive_field_pre_processor import \
    ReceptiveFieldPreProcessor
from src.model.deeplearn.dlrun.grid_subsampling_pre_processor import \
    GridSubsamplingPreProcessor
from src.model.deeplearn.dlrun.hierarchical_fps_pre_processorpp import \
    HierarchicalFPSPreProcessorPP
from src.utils.ptransf.receptive_field_hierarchical_sg import \
    ReceptiveFieldHierarchicalSG
import src.main.main_logger as LOGGING
import pyvl3dpp as vl3dpp
import tensorflow as tf
from tensorflow.lookup import StaticHashTable, KeyValueTensorInitializer
import numpy as np
import time

from src.utils.ptransf.receptive_field_hierarchical_fpspp import ReceptiveFieldHierarchicalFPSPP


# ---   CLASS   --- #
# ----------------- #
class HierarchicalSGPreProcessorPP(ReceptiveFieldPreProcessor):
    """
    :author: Alberto M. Esmoris Pena

    Preprocess the input dictionary of X (coordinates), F (features), and y
    (expected values) so it can be feed to a hierarchical sparse neural network
    such as sparse convolutional neural network.

    See :class:`.ReceptiveFieldPreProcessor`.
    See :class:`.ReceptiveFieldHierarchicalSG`.

    :ivar cell_size: See `size` of :class:`.ReceptiveFieldHierarchicalSG`.
    :vartype cell_size: float
    :ivar submanifold_window: See `w` of :class:`.ReceptiveFieldHierarchicalSG`.
    :vartype submanifold_window: :class:`np.ndarray`
    :ivar downsampling_window: See `wD` of
        :class:`.ReceptiveFieldHierarchicalSG`.
    :vartype downsampling_window: :class:`np.ndarray`
    :ivar downsampling_stride: See `sD` of
        :class:`.ReceptiveFieldHierarchicalSG`.
    :vartype downsampling_stride: :class:`np.ndarray`
    :ivar upsampling_window: See `wU` of
        :class:`.ReceptiveFieldHierarchicalSG`.
    :vartype upsampling_window: :class:`np.ndarray`
    :ivar upsampling_stride: See `sU` of
        :class:`.ReceptiveFieldHierarchicalSG`.
    :vartype upsampling_stride: :class:`np.ndarray`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialization/instantiation of a Hierarchical Sparse Grid
        pre-processor.

        :param kwargs: The key-word arguments for the
            Hierarchical Sparse Grid pre-processor.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.cell_size = kwargs.get('cell_size', None)
        self.submanifold_window = kwargs.get('submanifold_window', None)
        self.downsampling_window = kwargs.get('downsampling_window', None)
        self.downsampling_stride = kwargs.get('downsampling_stride', None)
        self.upsampling_window = kwargs.get('upsampling_window', None)
        self.upsampling_stride = kwargs.get('upsampling_stride', None)
        self.pre_processor = self
        # Validate attributes
        if self.cell_size is None:
            raise DeepLearningException(
                'HierarchicalSGPreProcessorPP cannot be instantiated '
                'without cell size.'
            )
        if self.submanifold_window is None:
            raise DeepLearningException(
                'HierarchicalSGPreProcessorPP cannot be instantiated '
                'without submanifold window.'
            )
        if self.downsampling_window is None:
            raise DeepLearningException(
                'HierarchicalSGPreProcessorPP cannot be instantiated '
                'without downsampling window.'
            )
        if self.downsampling_stride is None:
            raise DeepLearningException(
                'HierarchicalSGPreProcessorPP cannot be instantiated '
                'without downsampling stride.'
            )
        if self.upsampling_window is None:
            raise DeepLearningException(
                'HierarchicalSGPreProcessorPP cannot be instantiated '
                'without upsampling window.'
            )
        if self.upsampling_stride is None:
            raise DeepLearningException(
                'HierarchicalSGPreProcessorPP cannot be instantiated '
                'without upsampling stride.'
            )
        if self.cell_size is None or self.cell_size <= 0:
            raise DeepLearningException(
                'HierarchicalSGPreProcessorPP cannot be instantiated '
                'without a cell size strictly greater than zero. '
                f'Cell size {self.cell_size} was given.'
            )
        if len(self.submanifold_window) < 1:
            raise DeepLearningException(
                'HierarchicalSGPreProcessorPP cannot be instantiated without '
                'submanifold windows.'
            )
        if len(self.submanifold_window) != (len(self.downsampling_window)+1):
            raise DeepLearningException(
                'HierarchicalSGPreProcessorPP cannot be instantiated with '
                f'{len(self.submanifold_window)} submanifold windows and '
                f'{len(self.downsampling_window)} downsampling windows. '
                'The number of submanifold windows must be the number of '
                'downsampling windows plus one.'
            )
        if len(self.downsampling_window) != len(self.downsampling_stride):
            raise DeepLearningException(
                'HierarchicalSGPreProcessorPP cannot be instantiated with '
                f'{len(self.downsampling_window)} downsampling windows but '
                f'{len(self.downsampling_stride)} downsampling strides. '
                'The number of downsampling windows must be equal to the '
                'number of downsampling strides.'
            )
        if len(self.downsampling_window) != len(self.upsampling_window):
            raise DeepLearningException(
                'HierarchicalSGPreProcessorPP cannot be instantiated with '
                f'{len(self.downsampling_window)} downsampling windows but '
                f'{len(self.upsampling_window)} upsampling windows. '
                'The number of upsampling windows must be equal to the '
                'number of upsampling windows.'
            )
        if len(self.upsampling_window) != len(self.upsampling_stride):
            raise DeepLearningException(
                'HierarchicalSGPreProcessorPP cannot be instantiated with '
                f'{len(self.upsampling_window)} upsampling windows but '
                f'{len(self.upsampling_stride)} upsampling strides. '
                'The number of upsampling windows must be equal to the '
                'number of upsampling strides.'
            )
        # Convert some attributes to arrays (will prevent unnecessary copies)
        self.submanifold_window = np.array(
            self.submanifold_window, dtype=np.int32
        )
        self.downsampling_window = np.array(
            self.downsampling_window, dtype=np.int32
        )
        self.downsampling_stride = np.array(
            self.downsampling_stride, dtype=np.int32
        )
        self.upsampling_window = np.array(
            self.upsampling_window, dtype=np.int32
        )
        self.upsampling_stride = np.array(
            self.upsampling_stride, dtype=np.int32
        )

    # ---   RUN/CALL   --- #
    # -------------------- #
    def __call__(self, inputs):
        r"""
        Executes the pre-processing logic.

        :param inputs: A key-word input where the key "X" gives the input
            dataset and the "y" (OPTIONALLY) gives the reference values
            that can be used to fit/train a sparse hierarchical model.
        :type inputs: dict
        :return: (F, ...ht..., ...hDt..., ...hUt..., ...nt...). Where F are the
            input features for the first grid in the hierarchy, ht is the map
            whose keys (domain) are the index of active cells at depth t and
            whose values (co-domain) represent the sequential indexing of the
            active cells (e.g., to know what row from F corresponds with a
            given active cell at the first depth), hDt is the indexing vector
            that gives the index of the min vertex of the convolutional window
            at depth t that generates the corresponding active cell at depth
            t+1, hUt is the indexing vector that gives the index of the min
            vertex of the convolutional window at depth t+1 that generates the
            corresponding active cell at depth t, and nt is the number of
            axis-wise partitions along each axis at depth t.
        """
        LOGGING.LOGGER.info(
            'Generating hierarchical sparse grid receptive fields using the '
            'C++ extensions ...'
        )
        # Extract inputs
        start = time.perf_counter()
        X, F = inputs['X']
        if F.dtype != np.float32:
            F = F.astype(np.float32)
        y = inputs.get('y', None)
        if y is not None and y.dtype != np.int32:
            y = y.astype(np.int32)
        # Determine number of classes if not available
        ReceptiveFieldPreProcessor.num_classes_from_pwise_labels(self, y)
        # Purge old state before computing the new one
        self.purge_receptive_fields()
        # Prepare C++ call
        y = np.array([]) if y is None else y
        training_class_distribution = FurthestPointSubsamplingPreProcessorPP\
            .prepare_training_class_distribution(
                self.training_class_distribution
            )
        radii = FurthestPointSubsamplingPreProcessorPP.prepare_radii(
            self.neighborhood_spec
        )
        Xdtype = X.dtype
        if Xdtype == np.float32:
            cpp_f = vl3dpp.rf_dl_hsg_preproc_Xf_Ff_Is32_ys32
        else:
            cpp_f = vl3dpp.rf_dl_hsg_preproc_Xd_Ff_Is32_ys32
        # Call C++ to generate the receptive fields
        Fout, yout, h, hD, hU, n, A = cpp_f(
            X,
            F,
            y,
            self.num_classes,
            self.cell_size,
            self.submanifold_window,
            self.downsampling_window,
            self.downsampling_stride,
            self.upsampling_window,
            self.upsampling_stride,
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
            self.nthreads
        )
        # Transform h dictionaries to TensorFlow static hash tables
        h = [
            [
                [
                    HierarchicalFPSPreProcessorPP.optimize_indexing_memory(
                        np.array(list(hkt.keys()), dtype=np.int32)
                    ),
                    HierarchicalFPSPreProcessorPP.optimize_indexing_memory(
                        np.array(list(hkt.values()), dtype=np.int32)+1
                    )
                ]
                for hkt in hk
            ]
            for hk in h
        ]
        # Transform downsampling and upsampling maps to contain 1d vectors
        hD = [
            [
                HierarchicalFPSPreProcessorPP.optimize_indexing_memory(
                    hDkt.flatten()
                )
                for hDkt in hDk
            ]
            for hDk in hD
        ]
        hU = [
            [
                HierarchicalFPSPreProcessorPP.optimize_indexing_memory(
                    hUkt.flatten()
                )
                for hUkt in hUk
            ]
            for hUk in hU
        ]
        # Transform Fout so first row is ground vector (all zeroes)
        Fout = [
            np.vstack([np.zeros((1, Fouti.shape[1]), dtype=Fouti.dtype), Fouti])
            for Fouti in Fout
        ]
        # Transform A, yout to contain 1d vectors
        A = [Ai.flatten() for Ai in A]
        # Handle reference labels, if any
        if yout is None or len(yout) < 1:
            yout = None
        else:
            yout = [
                HierarchicalFPSPreProcessorPP.optimize_indexing_memory(
                    youti.flatten()
                )
                for youti in yout
            ]
        # Create receptive fields
        self.last_call_receptive_fields = []
        for i in range(len(h)):
            # Create i-th receptive field
            rf = ReceptiveFieldHierarchicalSG(
                cell_size=self.cell_size,
                submanifold_window=self.submanifold_window,
                downsampling_window=self.downsampling_window,
                downsampling_stride=self.downsampling_stride,
                upsampling_window=self.upsampling_window,
                upsampling_stride=self.upsampling_stride
            )
            rf.h = h[i]
            rf.hD = hD[i]
            rf.hU = hU[i]
            rf.n = n[i]
            rf.A = A[i]
            # Track receptive field
            self.last_call_receptive_fields.append(rf)
        # Report time
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'The C++ hierarchical sparse grid pre-processor '
            f'generated {len(h)} receptive fields '
            f'of depth {len(h[0])} each '
            f'from {X.shape[0]} points in {end-start:.3f} seconds.'
        )
        # Export support points if requested
        if inputs.get('plots_and_reports', True):
            sup_X = np.vstack([
                rfi.A + rfi.size/2.0
                for rfi in self.last_call_receptive_fields
            ])
            self.export_support_points(inputs, sup_X)
        # Prepare return
        pyout = [Fout, h, hD, hU, n, A]
        # Return with labels
        if yout is not None:
            return pyout, yout
        # Return without labels
        return pyout

    # ---  UTIL METHODS  --- #
    # ---------------------- #
    def reduce_labels(self, X, y, I=None):
        r"""
        Reduce the given labels :math:`\pmb{y} in \mathbb{Z}_{\geq 0}^{m}`
        to the receptive field labels
        :math:`\pmb{y}_{k} \in \mathbb{Z}_{\geq 0}^{R_k}`.

        :param X: The matrix representing the structure space of the original
            input point cloud (i.e., NOT a particular receptive field).
        :type X: :class:`np.ndarray`
        :param y: The labels of the original point cloud that must be reduced
            to the receptive fields.
        :type y: :class:`np.ndarray`
        :param I: Not used by hierarchical sparse grids.
        :type I: None
        :return: The reduced labels for each receptive field.
        :rtype: list
        """
        # Prepare computation of receptive field-wise labels
        # TODO Rethink : Size as a scalar (common to all RFs) ?
        size, A, n, hk, hv = [], [], [], [], []
        for rfi in self.last_call_receptive_fields:
            size.append(rfi.size)
            A.append(rfi.get_min_point())
            n.append(rfi.get_num_partitions()[:, 0])
            h = rfi.get_submanifold_maps()[0]
            hk.append(h[0])
            #hv.append(h[1]) # TODO Rethink : Should use this and apply -1 before
            hv.append(h[1]-1)  # TODO Rethink : Debug only to check SIGSEGV
        size = np.array(size)
        # Determine C++ function
        cpp_f = HierarchicalSGPreProcessorPP.find_cpp_reduce_label_function(
            X, y
        )
        # Validate C++ function
        if cpp_f is None:
            raise DeepLearningException(
                'HierarchicalSGPreProcessorPP failed to reduce labels '
                'because the C++ function could not be determined.'
            )
        # Compute receptive field-wise labels
        out = cpp_f(X, y, size, A, n, hk, hv, self.nthreads)
        return out

    @staticmethod
    def find_cpp_reduce_label_function(X, y):
        """
        Determine the C++ function that must be used to reduce the point-wise
        labels considering the data types of the input.

        :param X: The matrix representing the structure space of the original
            input point cloud (i.e., NOT a particular receptive field).
        :type X: :class:`np.ndarray`
        :param y: The labels of the original point cloud that must be reduced
            to the receptive fields.
        :type y: :class:`np.ndarray`
        :return: The C++ function for label reduction.
        """
        # Find types
        Xdtype = X.dtype
        ydtype = y.dtype
        # Determine function
        if Xdtype == np.float32:  # 32 bits structure space
            if ydtype == np.int8:  # 8 bits signed labels
                return vl3dpp.rf_sparse_reduce_label_mode_Xfs8s32
            elif ydtype == np.int16:  # 16 bits signed labels
                return vl3dpp.rf_sparse_reduce_label_mode_Xfs16s32
            elif ydtype == np.int32:  # 32 bits signed labels
                return vl3dpp.rf_sparse_reduce_label_mode_Xfs32s32
            elif ydtype == np.uint8:  # 8 bits unsigned labels
                return vl3dpp.rf_sparse_reduce_label_mode_Xfu8s32
            elif ydtype == np.uint16:  # 16 bits unsigned labels
                return vl3dpp.rf_sparse_reduce_label_mode_Xfu16s32
            elif ydtype == np.uint32:  # 32 bits unsigned labels
                return vl3dpp.rf_sparse_reduce_label_mode_Xfu32s32
        else:  # 64 bits structure space
            if ydtype == np.int8:  # 8 bits signed labels
                return vl3dpp.rf_sparse_reduce_label_mode_Xds8s32
            elif ydtype == np.int16:  # 16 bits signed labels
                return vl3dpp.rf_sparse_reduce_label_mode_Xds16s32
            elif ydtype == np.int32:  # 32 bits signed labels
                return vl3dpp.rf_sparse_reduce_label_mode_Xds32s32
            elif ydtype == np.uint8:  # 8 bits unsigned labels
                return vl3dpp.rf_sparse_reduce_label_mode_Xdu8s32
            elif ydtype == np.uint16:  # 16 bits unsigned labels
                return vl3dpp.rf_sparse_reduce_label_mode_Xdu16s32
            elif ydtype == np.uint32:  # 32 bits unsigned labels
                return vl3dpp.rf_sparse_reduce_label_mode_Xdu32s32
        # No adequate function was found
        return None

    # ---   SUPPORT POINTS EXPORT   --- #
    # --------------------------------- #
    def _export_support_points(self, sup_X, path):
        """
        See :class:`.ReceptiveFieldPreProcessor`,
        :meth:`receptive_field_pre_processor.ReceptiveFieldPreProcessor._export_support_points`, and
        :meth:`GridSubsamplingPreProcessor.support_points_to_file`.
        """
        return GridSubsamplingPreProcessor.support_points_to_file(sup_X, path)

    # ---   OTHER METHODS   --- #
    # ------------------------- #
    def overwrite_pretrained_model(self, spec):
        """
        Assist the :meth:`model.Model.overwrite_pretrained_model` method
        through assisting the
        :meth:`architecture.Architecture.overwrite_pretrained_model` method.

        :param spec: The key-word specification containing the model's
            arguments.
        :type spec: dict
        """
        # Overwrite from parent
        super().overwrite_pretrained_model(spec)
        spec_keys = spec.keys()
        # Overwrite the attributes of the hierarchical SG pre-processor
        if 'cell_size' in spec_keys:
            self.cell_size = spec['cell_size']
        if 'submanifold_window' in spec_keys:
            self.submanifold_window = np.array(spec['submanifold_window'])
        if 'downsampling_window' in spec_keys:
            self.downsampling_window = np.array(spec['downsampling_window'])
        if 'downsampling_stride' in spec_keys:
            self.downsampling_stride = np.array(spec['downsampling_stride'])
        if 'upsampling_window' in spec_keys:
            self.upsampling_window = np.array(spec['upsampling_window'])
        if 'upsampling_stride' in spec_keys:
            self.upsampling_stride = np.array(spec['upsampling_stride'])

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def __getstate__(self):
        """
        Method to be called when saving the serialized hierarchical sparse grid
        receptive field pre-processor.

        See :meth:`.ReceptiveFieldPreProcessor.__getstate__`.

        :return: The state's dictionary of the object.
        :rtype: dict
        """
        # Obtain parent's state
        state = super().__getstate__()
        # Update state
        state['cell_size'] = self.cell_size
        state['submanifold_window'] = self.submanifold_window
        state['downsampling_window'] = self.downsampling_window
        state['downsampling_stride'] = self.downsampling_stride
        state['upsampling_window'] = self.upsampling_window
        state['upsampling_stride'] = self.upsampling_stride
        # Return
        return state

    def __setstate__(self, state):
        """
        Method to be called when loading and deserializing a previously
        serialized hierarchical sparse grid pre-processor.

        See :meth:`ReceptiveFieldPreProcessor.__setstate__`.

        :param state: The state's dictionary of the saved hierarchical
            sparse grid pre-processor.
        :type state: dict
        :return: Nothing, but modifies the internal state of the object.
        """
        # Call parent
        super().__setstate__(state)
        # Assign member attributes from state
        self.cell_size = state['cell_size']
        self.submanifold_window = state['submanifold_window']
        self.downsampling_window = state['downsampling_window']
        self.downsampling_stride = state['downsampling_stride']
        self.upsampling_window = state['upsampling_window']
        self.upsampling_stride = state['upsampling_stride']
