# ---   IMPORTS   --- #
# ⨪------------------ #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.layer.layer import Layer
import src.main.main_logger as LOGGING
import tensorflow as tf
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class ShadowActivationLayer(Layer):
    """
    Shadow version of an activation layer, i.e., it works with tensors with
    padding, where the padding is used to represent shadow
    values/points/cells/elements, i.e., those that must not have an impact
    on the computations.

    See :class:`tf.Tensor` and :class:`tf.keras.layers.Activation`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, activation, act=None, offset=0, **kwargs):
        """
        See :class:`.Layer` and :meth:`layer.Layer.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Instantiate underlying Activation layer
        self.act = act
        kwargs_act = dict(kwargs)
        if 'name' in kwargs_act:
            kwargs_act['name'] = f'RAG_{kwargs_act["name"]}'
        if self.act is None:
            self.act = tf.keras.layers.Activation(activation, **kwargs_act)
        self.offset = offset

    # ---   LAYER METHODS   --- #
    # ------------------------- #
    def build(self, dim_in):
        """
        Build the underlying :class:`tf.keras.layers.Activation` layer.

        See :class:`.Layer`, :meth:`layer.Layer.build`, and
        :class:`tf.keras.layers.Activation`.
        """
        # Call parent's build
        super().build(dim_in)
        # Build underlying Activation layer
        self.act.build(dim_in)

    def call(self, inputs):
        """
        Call the shadow Activation layer.

        See :class:`.Layer` and :meth:`layer.Layer.call`.
        """
        def activate(input):
            x, start = input
            start = tf.squeeze(start)
            xdim = len(tf.shape(x))
            if xdim == 1:
                return tf.pad(
                    self.act(x[start+self.offset:]),
                    [[start + self.offset, 0]],
                    "CONSTANT",
                    constant_values=0
                )
            elif xdim == 2:
                return tf.pad(
                    self.act(x[start+self.offset:]),
                    [[start + self.offset, 0], [0, 0]],
                    "CONSTANT",
                    constant_values=0
                )
            return tf.pad(
                self.act(x[start + self.offset:]),
                [
                    [start + self.offset, 0],
                    [0, 0],
                    [0, 0]
                ],
                "CONSTANT",
                constant_values=0
            )
        return tf.map_fn(
            activate,
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
            'act': tf.keras.layers.serialize(self.act)
        })
        # Return updated config
        return config

    @classmethod
    def from_config(cls, config):
        """Use given config data to deserialize the layer"""
        # Instantiate layer
        act = tf.keras.layers.deserialize(config['act'])
        config['act'] = act
        act = cls(**config)
        # Return deserialized layer
        return act
