# ---   IMPORTS   --- #
# ------------------- #
import src.model.deeplearn.dlrun.furthest_point_subsampling_post_processorpp
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.dlrun.hierarchical_fps_post_processor import \
    HierarchicalFPSPostProcessor
from src.model.deeplearn.dlrun.furthest_point_subsampling_post_processorpp \
    import FurthestPointSubsamplingPostProcessorPP


# ---   CLASS   --- #
# ----------------- #
class HierarchicalFPSPostProcessorPP(HierarchicalFPSPostProcessor):
    """
    :author: Alberto M. Esmoris Pena

    C++ version of :class:`.HierarchicalFPSPostProcessor`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, hfps_preproc, **kwargs):
        """
        C++ version of :meth:`.HierarchicalFPSPostProcessor.__init__`.
        """
        # Call parent's init
        super().__init__(hfps_preproc, **kwargs)

    # ---   RUN/CALL   --- #
    # -------------------- #
    def post_process(self, inputs, reducer):
        """
        C++ version of
        :meth:`.FurthestPointSubsamplingPostProcessor.post_process`.
        """
        # Extract inputs
        rf = self.hfps_preproc.last_call_receptive_fields
        I = self.hfps_preproc.last_call_neighborhoods
        nthreads = self.hfps_preproc.nthreads
        X = inputs['X']  # The original point cloud (before receptive field)
        z_reduced = inputs['z']  # Softmax scores reduced to receptive field
        num_classes = z_reduced.shape[-1]
        # Determine C++ function to be called
        Mdtype = self.hfps_preproc.last_call_receptive_fields[0].NUs[0].dtype
        Idtype = I[0].dtype
        zdtype = z_reduced.dtype
        cpp_f = FurthestPointSubsamplingPostProcessorPP.find_cpp_postproc_fun(
            Mdtype, Idtype, zdtype
        )
        cpp_reduction_type = \
            FurthestPointSubsamplingPostProcessorPP.find_cpp_reduction_type(
                reducer
            )
        cpp_f_extra_args = \
            FurthestPointSubsamplingPostProcessorPP.extract_cpp_extra_args(
                reducer
            )
        # Transform each prediction by propagation and point-wise reduction
        return cpp_f(
            cpp_reduction_type,
            X.shape[0],
            z_reduced,
            [rfi.NUs[0] for rfi in rf],
            I,
            *cpp_f_extra_args,
            nthreads
        )
