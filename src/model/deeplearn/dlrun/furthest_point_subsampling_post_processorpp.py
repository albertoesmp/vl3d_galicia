# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.dlrun.furthest_point_subsampling_post_processor \
    import  FurthestPointSubsamplingPostProcessor
from src.utils.preds.mean_pred_reduce_strategy import MeanPredReduceStrategy
from src.utils.preds.sum_pred_reduce_strategy import SumPredReduceStrategy
from src.utils.preds.max_pred_reduce_strategy import MaxPredReduceStrategy
from src.utils.preds.entropic_pred_reduce_strategy import \
    EntropicPredReduceStrategy
import pyvl3dpp as vl3dpp
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class FurthestPointSubsamplingPostProcessorPP(
    FurthestPointSubsamplingPostProcessor
):
    """
    :author: Alberto M. Esmoris Pena

    C++ implementation of the :class:`.FurthestPointSubsamplingPostProcessor`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, fps_preproc, **kwargs):
        """
        C++ version of :meth:`.FurthestPointSubsamplingPostProcessor.__init__`.`
        """
        # Call parent's init
        super().__init__(fps_preproc, **kwargs)

    # ---   RUN/CALL   --- #
    # -------------------- #
    def post_process(self, inputs, reducer):
        """
        C++ version of
        :meth:`.FurthestPointSubsamplingPostProcessor.post_process`.
        """
        # Extract inputs
        rf = self.fps_preproc.last_call_receptive_fields
        I = self.fps_preproc.last_call_neighborhoods
        nthreads = self.fps_preproc.nthreads
        X = inputs['X']  # The original point cloud (before receptive field)
        z_reduced = inputs['z']  # Softmax scores reduced to receptive field
        num_classes = z_reduced.shape[-1]
        # Determine C++ function to be called
        Mdtype = self.fps_preproc.last_call_receptive_fields[0].M.dtype
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
            [rfi.M for rfi in rf],
            I,
            *cpp_f_extra_args,
            nthreads
        )

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    @staticmethod
    def find_cpp_postproc_fun(Mdtype, Idtype, zdtype):
        """
        Determine the C++ function that must be used to post-process the output
        of the neural network back to the original point cloud.
        """
        cpp_f = None
        if zdtype == np.float32:  # 32 bits per probability
            if Mdtype == np.uint8:  # 8 bits per encoding index
                if Idtype == np.uint8:  # 8 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_fu8u8
                elif Idtype == np.uint16:  # 16 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_fu8u16
                elif Idtype == np.uint32:  # 32 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_fu8u32
                else:  # 64 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_fu8u64
            elif Mdtype == np.uint16:  # 16 bits per index
                if Idtype == np.uint8:  # 8 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_fu16u8
                elif Idtype == np.uint16:  # 16 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_fu16u16
                elif Idtype == np.uint32:  # 32 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_fu16u32
                else:  # 64 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_fu16u64
            elif Mdtype == np.uint32:  # 32 bits per index
                if Idtype == np.uint8:  # 8 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_fu32u8
                elif Idtype == np.uint16:  # 16 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_fu32u16
                elif Idtype == np.uint32:  # 32 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_fu32u32
                else:  # 64 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_fu32u64
            else:  # 64 bits per index
                if Idtype == np.uint8:  # 8 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_fu64u8
                elif Idtype == np.uint16:  # 16 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_fu64u16
                elif Idtype == np.uint32:  # 32 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_fu64u32
                else:  # 64 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_fu64u64
        else:  # 64 bits per probability
            if Mdtype == np.uint8:  # 8 bits per encoding index
                if Idtype == np.uint8:  # 8 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_du8u8
                elif Idtype == np.uint16:  # 16 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_du8u16
                elif Idtype == np.uint32:  # 32 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_du8u32
                else:  # 64 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_du8u64
            elif Mdtype == np.uint16:  # 16 bits per index
                if Idtype == np.uint8:  # 8 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_du16u8
                elif Idtype == np.uint16:  # 16 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_du16u16
                elif Idtype == np.uint32:  # 32 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_du16u32
                else:  # 64 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_du16u64
            elif Mdtype == np.uint32:  # 32 bits per index
                if Idtype == np.uint8:  # 8 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_du32u8
                elif Idtype == np.uint16:  # 16 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_du32u16
                elif Idtype == np.uint32:  # 32 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_du32u32
                else:  # 64 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_du32u64
            else:  # 64 bits per index
                if Idtype == np.uint8:  # 8 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_du64u8
                elif Idtype == np.uint16:  # 16 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_du64u16
                elif Idtype == np.uint32:  # 32 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_du64u32
                else:  # 64 bits per decoding index
                    cpp_f = vl3dpp.rf_dl_fps_postproc_mean_du64u64
        if cpp_f is None:  # Right now, it cant be none (but this could change)
            raise DeepLearningException(
                'FurthestPointSubsamplingPostProcessorPP failed to post '
                'process because no valid C++ function was found.'
            )
        return cpp_f

    @staticmethod
    def find_cpp_reduction_type(reducer):
        """
        Determine the C++ reduction strategy that must be used to post-process
        the output of the neural network back to the original point cloud.
        """
        # Determine reduction type
        if reducer is not None and reducer.reduce_strategy is not None:
            reduce_strategy = type(reducer.reduce_strategy)
            if reduce_strategy == SumPredReduceStrategy:
                return "sum_reduce"
            elif reduce_strategy == MaxPredReduceStrategy:
                return "max_reduce"
            elif reduce_strategy == EntropicPredReduceStrategy:
                return "entropic_reduce"
        # Default reduction type
        return "mean_reduce"

    @staticmethod
    def extract_cpp_extra_args(reducer):
        """
        Extract the extra arguments to be passed to the cpp function.
        See
        :meth:`FurthestPointSubsamplingPostProcessorPP.find_cpp_postproc_fun`
        .

        :return: The extra arguments are returned as a list that always follows
            the same order so it can be easily understood by the C++ code.
        :rtype list:
        """
        # Extract args
        min_clip_value = getattr(reducer, 'min_clip_value', 1e-6)
        # Return as list
        return [min_clip_value]