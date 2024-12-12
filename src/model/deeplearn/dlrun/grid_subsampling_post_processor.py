# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
import src.main.main_logger as LOGGING
from src.main.main_config import VL3DCFG
import numpy as np
import joblib
from joblib.externals.loky import get_reusable_executor
import time
import gc


# ---   CLASS   --- #
# ----------------- #
class GridSubsamplingPostProcessor:
    """
    :author: Alberto M. Esmoris Pena

    Postprocess an input in the grid subsampling space back to the
    original space before the subsampling.

    See :class:`.GridSubsamplingPreProcessor`.

    :ivar gs_preproc: The preprocessor that generated the grid subsampling
        that must be reverted by the post-processor.
    :vartype gs_preproc: :class:`.GridSubsamplingPreProcessor`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, gs_preproc, **kwargs):
        """
        Initialization/instantiation of a Grid Subsampling post-processor.

        :param kwargs: The key-word arguments for the
            GridSubsamplingPostProcessor.
        """
        # Assign attributes
        self.gs_preproc = gs_preproc
        if self.gs_preproc is None:
            raise DeepLearningException(
                'GridSubsamplingPostProcessor needs the corresponding '
                'GridSubsamplingPreProcessor.'
            )

    # ---   RUN/CALL   --- #
    # -------------------- #
    def __call__(self, inputs, reducer=None):
        """
        Executes the post-processing logic.

        :param inputs: A key-word input where the key "X" gives the coordinates
            of the points in the original point cloud. Also, the key "z" gives
            the predictions computed on a receptive field of :math:`R` points
            that must be propagated back to the :math:`m` points of the
            original point cloud.
        :type inputs: dict
        :param reducer: The prediction reducer for the post-processor, if any.
        :type reducer: :class:`.PredictionReducer`
        :return: The :math:`m` point-wise predictions derived from the
            :math:`R` input predictions on the receptive field.
        """
        start = time.perf_counter()
        _inputs = inputs
        if isinstance(inputs['X'], list):
            _inputs = {
                'X': inputs['X'][0],
                'z': inputs['z']
            }
        z = GridSubsamplingPostProcessor.post_process(
            _inputs,
            self.gs_preproc.last_call_receptive_fields,
            self.gs_preproc.last_call_neighborhoods,
            nthreads=self.gs_preproc.nthreads,
            reducer=reducer
        )
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'The grid subsampling post processor generated {len(z)} '
            f'propagations from {len(inputs["z"][0])} reduced predictions '
            f'for each of the {len(inputs["z"])} GS receptive fields '
            f'in {end-start:.3f} seconds.'
        )
        return z

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    @staticmethod
    def pwise_reduce(npoints, nvars, I, v_propagated):
        """
        Compute a point-wise reduction of propagated values with overlapping.
        In other words, this method can be used to reduce values computed
        on overlapping neighborhoods so there is potentially more than one
        value for the same variable of the same point. The reduction consists
        of computing the mean value.


        :param npoints: The number of points.
        :param nvars: The number of considered point-wise variables.
        :param I: The list of neighborhoods. I[i] is the list of indices
            corresponding to the points composing the neighborhood i.
        :param v_propagated: The values to be point-wise reduced. They often
            come from a propagation operation computed on a receptive field,
            thus the name.
        :return: The reduced v vector with a single value for the same variable
            of the same point.
        :rtype: :class:`np.ndarray`
        """
        count = np.zeros(npoints, dtype=int)
        utype = v_propagated[0].dtype
        u = np.zeros((npoints, nvars), dtype=utype) \
            if len(v_propagated[0].shape) > 1 \
            else np.zeros(npoints, dtype=utype)
        for i, v_prop_i in enumerate(v_propagated):
            u[I[i]] += v_prop_i
            count[I[i]] += 1
        non_zero_mask = count != 0
        u[non_zero_mask] = \
            u[non_zero_mask] / count[non_zero_mask] if len(u.shape) < 2 \
            else (u[non_zero_mask].T/count[non_zero_mask]).T
        # Return
        return u

    @staticmethod
    def post_process(inputs, rf, I, nthreads=1, reducer=None):
        """
        Computes the post-processing logic. The method is used to aid the
        :meth:`grid_subsampling_post_processor.GridSubsamplingPostProcessor.__call__`
        method.

        :param inputs: A key-word input where the key "X" gives the coordinates
            of the points in the original point cloud. Also, the key "z" gives
            the predictions computed on a receptive field of :math:`R` points
            that must be propagated back to the :math:`m` points of the
            original point cloud.
        :type inputs: dict
        :param rf: The receptive fields to compute the propagations. See
            :class:`.ReceptiveField` and :class:`.ReceptiveFieldGS`.
        :type rf: list
        :param I: The list of neighborhoods, where each neighborhood is given
            as a list of indices.
        :type I: list
        :param nthreads: The number of threads for parallel computing.
        :type nthreads: int
        :param reducer: The prediction reducer for the post-processor, if any.
        :type reducer: :class:`.PredictionReducer`
        :return: The :math:`m` point-wise predictions derived from the
            :math:`R` input predictions on the receptive field.
        """
        # Extract inputs
        X = inputs['X']  # The original point cloud (before receptive field)
        z_reduced = inputs['z']  # Softmax scores reduced to receptive field
        num_classes = z_reduced.shape[-1]
        # Transform each prediction by propagation
        max_classes_per_reduction = VL3DCFG['MODEL']['ReceptiveField'].get(
            'max_classes_per_reduction',
            16
        )
        num_spans = int(np.ceil(num_classes/max_classes_per_reduction))
        reductions = np.zeros(
            [X.shape[0], z_reduced.shape[-1]],
            dtype=z_reduced.dtype
        )
        for class_span_idx in range(num_spans):
            # Determine class span (interval [a, b))
            class_start = class_span_idx*max_classes_per_reduction
            class_end = (1+class_span_idx)*max_classes_per_reduction
            class_end = min(num_classes, class_end)
            num_classes_in_span = class_end-class_start
            # Propagate reductions
            z_propagated = joblib.Parallel(n_jobs=nthreads)(
                joblib.delayed(
                    rfi.propagate_values
            )(
                    z_reduced[i, :, class_start:class_end],
                    reduce_strategy='mean'
                )
                for i, rfi in enumerate(rf)
            )
            # Reduce many point-wise predictions through given reducer
            if reducer is not None:
                if z_reduced.shape[-1] == 1:
                    reductions[:, class_start:class_end] = reducer.reduce(
                        X.shape[0], num_classes_in_span, z_propagated, I
                    ).reshape(-1, 1)
                else:
                    reduced = reducer.reduce(
                        X.shape[0], num_classes_in_span, z_propagated, I
                    )
                    if len(reduced.shape) == 1:
                        reduced = reduced.reshape(-1, 1)
                    reductions[:, class_start:class_end] = reduced
            # Reduce many point-wise predictions through default function
            else:
                if z_reduced.shape[-1] == 1:
                    reductions[:, class_start:class_end] = \
                        GridSubsamplingPostProcessor.pwise_reduce(
                            X.shape[0], num_classes_in_span, I, z_propagated
                        ).reshape(-1, 1)
                else:
                    reductions[:, class_start:class_end] = \
                        GridSubsamplingPostProcessor.pwise_reduce(
                            X.shape[0], num_classes_in_span, I, z_propagated
                        )
            # Release propagated values (no longer needed)
            z_propagated = None
            get_reusable_executor().shutdown(wait=True)  # Release loky workers
            gc.collect()
        # Return point-wise reduced probabilities
        return reductions
