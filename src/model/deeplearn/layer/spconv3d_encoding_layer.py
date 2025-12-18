# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.layer.layer import Layer
from src.model.deeplearn.layer.submanifold_spconv3d_layer import \
    SubmanifoldSpConv3DLayer
from src.model.deeplearn.layer.downsampling_spconv3d_layer import \
    DownsamplingSpConv3DLayer
from src.model.deeplearn.layer.shadow_conv1d_layer import ShadowConv1DLayer
from src.model.deeplearn.layer.shadow_batch_normalization_layer import \
    ShadowBatchNormalizationLayer
from src.model.deeplearn.layer.shadow_activation_layer import \
    ShadowActivationLayer
import tensorflow as tf
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class SpConv3DEncodingLayer(Layer):
    r"""
    :author: Alberto M. Esmoris Pena
    TODO Rethink : Doc
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        See :class:`.Layer` and :meth:`layer.Layer.__init__`.
        """
        # Call paret's init
        super().__init__(**kwargs)
        # Assign attributes
        # TODO Rethink : Implement
