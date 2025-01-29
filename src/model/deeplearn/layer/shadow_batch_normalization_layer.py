# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.layer.layer import Layer
import src.main.main_logger as LOGGING
import tensorflow as tf
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class ShadowBatchNormalizationLayer(Layer):
    """
    Shadow version of a batch normalization layer, i.e., it works with tensors
    with padding, where the padding is used to represent shadow
    values/points/cells/elements, i.e., those that must not have an impact
    on the computations.

    See :class:`tf.Tensor` and
    :class:`tf.keras.layers.BatchNormalization`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(
        self,
        bn=None,
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        center=True,
        scale=True,
        beta_initializer='zeros',
        gamma_initializer='ones',
        moving_mean_initializer='ones',
        moving_variance_initializer='ones',
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        synchronized=False,
        offset=0,
        **kwargs
    ):
        """
        See :class:`.Layer` and :meth:`layer.Layer.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Instantiate underlying BatchNormalization layer
        self.bn = bn
        if self.bn is None:
            self.bn = tf.keras.layers.BatchNormalization(
                axis=axis,
                momentum=momentum,
                epsilon=epsilon,
                center=center,
                scale=scale,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
                moving_mean_initializer=moving_mean_initializer,
                moving_variance_initializer=moving_variance_initializer,
                beta_regularizer=beta_regularizer,
                gamma_regularizer=gamma_regularizer,
                beta_constraint=beta_constraint,
                gamma_constraint=gamma_constraint,
                synchronized=synchronized,
                name=f'RAG_{kwargs.get("name", "undBN")}'
            )
        self.offset = offset

    # ---   LAYER METHODS   --- #
    # ------------------------- #
    def build(self, dim_in):
        """
        Build the underlying :class:`tf.keras.layers.BatchNormalization` layer.

        See :class:`.Layer`, :meth:`layer.Layer.build`, and
        :class:`tf.keras.layers.BatchNormalization`.
        """
        # Call parent's build
        super().build(dim_in)
        # Build underlying BatchNormalization layer
        self.bn.build(dim_in[0][1:])

    def call(self, inputs, training=False, mask=False):
        """
        Call the shadow BatchNormalization layer.

        See :class:`.Layer` and :meth:`layer.Layer.call`.
        """
        def bn_tensor(input):
            x, start = input
            start = tf.squeeze(start)
            if len(tf.shape(x)) == 1:
                return tf.pad(
                    self.bn(x[start+self.offset:], training=training),
                    [[start + self.offset, 0]],
                    "CONSTANT",
                    constant_values=0
                )
            else:
                return tf.pad(
                    self.bn(x[start+self.offset:], training=training),
                    [[start+self.offset, 0], [0, 0]],
                    "CONSTANT",
                    constant_values=0
                )
        return tf.map_fn(
            bn_tensor,
            inputs,
            fn_output_signature=tf.TensorSpec(
                shape=[None for k in range(1, len(tf.shape(inputs[0])))],
                dtype=tf.dtypes.float32
            )
        )

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def get_config(self):
        """Return necessary data to serialize the layer"""
        # Call parent's config
        config = super().get_config()
        # Update config with custom attributes
        config.update({
            # Base attributes
            'bn': tf.keras.layers.serialize(self.bn)
        })
        # Return updated config
        return config

    @classmethod
    def from_config(cls, config):
        """Use given config data to deserialize the layer"""
        # Instantiate layer
        bn = tf.keras.layers.deserialize(config['bn'])
        config['bn'] = bn
        bnl = cls(**config)
        # Return deserialized layer
        return bnl
