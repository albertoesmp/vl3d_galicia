# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
import src.main.main_logger as LOGGING
import pyvl3dpp as vl3dpp
import numpy as np
import time

# ---   CLASS   --- #
# ----------------- #
class HierarchicalSGPostProcessorPP:
    """
    :author: Alberto M. Esmoris Pena

    Postprocess the data from the first level of the SG hierarchy back to the
    original space.

    See :class:`.HierarchicalSGPreProcessor`.

    :ivar hsg_preproc: The preprocessor that generated the hierarchical sparse
        grid that must be reverted by the post-processor.
    :vartype hfps_preproc: :class:`.HierarchicalSGPreProcessor`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, hsg_preproc, **kwargs):
        """
        Initialization/instantiation of a hierarchical SG post-processor.

        :param hsg_preproc: The corresponding hierarchical SG pre-processor.
        :param kwargs: The key-word arguments for the
            HierarchicalSGPostProcessor.
        """
        # Assign attributes
        self.hsg_preproc = hsg_preproc
        if self.hsg_preproc is None:
            raise DeepLearningException(
                'HierarchicalSGPostProcessor needs the '
                'corresponding HierarchicalSGPreProcessor.'
            )

    # ---   RUN/CALL   --- #
    # -------------------- #
    def __call__(self, inputs, reducer=None):
        """
        Executes the post-processing logic.

        :param inputs: A key-word input where the key "X" gives the coordinates
            of the points in the original point cloud. Also, the key "z" gives
            a list where each element contains the predictions computed on a
            sparse grid receptive field.
        :type inputs: dict
        :param reducer: The prediction reducer for the post-processor, if any.
        :type reducer: :class:`.PredictionReducer`
        :return: The :math:`m` point-wise predictions derived from the
            :math:`R` input predictions on the receptive field.
        :rtype: :class:`np.ndarray`
        """
        start = time.perf_counter()
        _inputs = inputs
        if isinstance(inputs['X'], list):
            _inputs = {
                'X': inputs['X'][0],
                'z': inputs['z']
            }
        z = self.post_process(_inputs, reducer)
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'The hierarchical SG post processor generated {len(z)} '
            f'propagations from the reduced predictions '
            f'for each of the {len(inputs["z"])} SG receptive fields '
            f'in {end-start:.3f} seconds.'
        )
        return z

    def post_process(self, inputs, reducer):
        """
        Assists the :meth:`.HierarchicalSGPostProcessor.__call__`
        providing the post-process logic itself.
        """
        # Extract inputs
        rf = self.hsg_preproc.last_call_receptive_fields
        nthreads = self.hsg_preproc.nthreads
        X = inputs['X']  # The original point cloud (before receptive field)
        z_reduced = inputs['z']  # Softmax scores reduced to receptive field
        num_classes = z_reduced[0].shape[-1]
        # Determine C++ function to be called
        Xtype = X.dtype
        ztype = z_reduced[0].dtype
        cpp_f = HierarchicalSGPostProcessorPP.find_cpp_postproc_fun(
            Xtype, ztype
        )
        # Transform each prediction by propagation and point-wise reduction
        return cpp_f(
            "mean_reduce",  # cpp_reduction_type@HierarchicalFPSPostProcessorPP
            X,
            z_reduced,
            rf[0].size,
            [rfi.A for rfi in rf],
            [rfi.n[:, 0] for rfi in rf],
            [
                rfi.get_submanifold_map_as_dict(0, without_ground=True)
                for rfi in rf
            ],
            0,  # cpp_f_extra_args@HierarchicalFPSPostProcessorPP
            nthreads
        )

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    @staticmethod
    def find_cpp_postproc_fun(Xtype, ztype):
        """
        Determine the C++ function that must be used to post-process the output
        of the neural network back to the original point cloud.
        """
        cpp_f = None
        if Xtype == np.float32:  # 32 bits structure space
            if ztype == np.float32:  # 32 bits probabilities
                cpp_f = vl3dpp.rf_dl_sg_postproc_mean_Xf_Ff_Is32
            else:  # 64 bits probabilities
                cpp_f = vl3dpp.rf_dl_sg_postproc_mean_Xf_Fd_Is32
        else:  # 64 bits structure space
            if ztype == np.float32:  # 32 bits probabilities
                cpp_f = vl3dpp.rf_dl_sg_postproc_mean_Xd_Ff_Is32
            else:  # 64 bits probabilities
                cpp_f = vl3dpp.rf_dl_sg_postproc_mean_Xd_Fd_Is32
        if cpp_f is None:  # Right now, it cant be none (but this could change)
            raise DeepLearningException(
                'HierarchicalSGPostProcessorPP failed to post process because '
                'no valid C++ function was found.'

            )
        return cpp_f