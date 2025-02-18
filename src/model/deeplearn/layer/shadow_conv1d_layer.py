# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.layer.layer import Layer
import src.main.main_logger as LOGGING
import tensorflow as tf
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class ShadowConv1DLayer(Layer):
    r"""
    Shadow version of a 1D convolutional layer, i.e., it works with tensors with
    padding, where the padding is used to represent shadow
    values/points/cells/elements, i.e., those that must not have an impact
    on the computations.

    See :class:`tf.Tensor` and :class:`tf.keras.layers.Conv1D`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(
        self,
        filters,
        kernel_size,
        conv1D=None,
        strides=1,
        padding="valid",
        data_format="channels_last",
        dilation_rate=1,
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        offset=0,
        **kwargs
    ):
        """
        See :class:`.Layer` and :meth:`layer.Layer.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Instantiate underlying Conv1D layer
        self.conv1D = conv1D
        if self.conv1D is None:
            self.conv1D = tf.keras.layers.Conv1D(
                filters,
                kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate,
                groups=groups,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                name=f'SHAD_{kwargs.get("name", "undConv1D")}'
            )
        self.offset = offset
        # Warn about unexpected configurations
        unitary_kernel_size = (
            kernel_size == 1 if isinstance(kernel_size, int) else
            kernel_size[0] if len(kernel_size) == 1 else False
        )
        unitary_stride = (
            strides == 1 if isinstance(strides, int) else
            strides[0] if len(strides) == 1 else False
        )
        if not unitary_kernel_size or not unitary_stride:
            LOGGING.LOGGER.warning(
                'ShadowConv1DLayer has been initialized with '
                f'window size {kernel_size} and stride {strides}.\n'
                'The ShadowConv1DLayer is tested to be used as a SharedMLP '
                'i.e., with window size 1 and stride 1.'
                'Other uses might lead to errors. USE WITH CAUTION AND ON '
                'YOUR OWN RESPONSIBILITY.'
            )

    # ---   LAYER METHODS   --- #
    # ------------------------- #
    def build(self, dim_in):
        """
        Build the underlying :class:`tf.keras.layers.Conv1D` layer.

        See :class:`.Layer`, :meth:`layer.Layer.build`, and
        :class:`tf.keras.layers.Conv1D`.
        """
        # Call parent's build
        super().build(dim_in)
        # Build underlying Conv1D layer
        self.conv1D.build(dim_in[0])

    def call(self, inputs):
        """
        Call the shadow Conv1D layer.

        See :class:`.Layer` and :meth:`layer.Layer.call`.
        """
        def handle_padding(input):
            x, start = input
            start = tf.squeeze(start)
            return tf.pad(
                x[start+self.offset:],
                [[start+self.offset, 0], [0, 0]],
                "CONSTANT",
                constant_values=0
            )
        return tf.map_fn(
            fn=handle_padding,
            elems=[self.conv1D(inputs[0]), inputs[1]],
            fn_output_signature=tf.TensorSpec(
                shape=(None, self.conv1D.filters),
                dtype=tf.dtypes.float32
            )
        )

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def get_config(self):
        """Return necessary data to serialize the layer"""
        # Call parent's config
        config = super().get_config()
        # update config with custom attributes
        config.update({
            # Base attributes
            'offset': self.offset,
            'conv1D': tf.keras.layers.serialize(self.conv1D)
        })
        # Return updated config
        return config

    @classmethod
    def from_config(cls, config):
        """Use given config data to deserialize the layer"""
        # Instantiate layer
        conv1D = tf.keras.layers.deserialize(config['conv1D'])
        config['conv1D'] = conv1D
        c1Dl = cls(
            conv1D.filters,
            conv1D.kernel_size,
            **config
        )
        # Return deserialized layer
        return c1Dl
