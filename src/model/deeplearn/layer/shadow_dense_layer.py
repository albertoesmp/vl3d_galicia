# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.layer.layer import Layer
import tensorflow as tf
import time  # TODO Remove : Debug only


# ---   CLASS   --- #
# ----------------- #
class ShadowDenseLayer(Layer):
    r"""
    Shadow version of a dense layer, i.e., it works with tensors with
    padding, where the padding is used to represent shadow
    values/points/cells/elements, i.e., those that must not have an impact
    on the computations.


    See :class:`tf.Tensor` and :class:`tf.keras.layers.Dense`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(
            self,
            units,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            offset=0,
            built_kernel=False,
            built_bias=False,
            **kwargs
    ):
        """
        See :class:`.Layer` and :meth:`layer.Layer.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(
            activity_regularizer
        )
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.offset = offset
        # Attributes initialized to None (derived when building)
        self.kernel = None
        self.built_kernel = built_kernel
        self.bias = None
        self.built_bias = built_bias

    # ---   LAYER METHODS   --- #
    # ------------------------- #
    def build(self, dim_in):
        """
        Build the kernel and the bias of the layer.

        See :class:`.Layer` and :meth:`layer.Layer.build`.
        """
        # Call parent's build
        super().build(dim_in)
        # Find input space dimensionality (e.g., number of features per element)
        nf = dim_in[0][-1]
        # Build the kernel
        if not self.built_kernel:
            self.kernel = self.add_weight(
                shape=(nf, self.units),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype='float32',
                trainable=True,
                name='kernel'
            )
            self.built_kernel = True
        # Validate the kernel
        if self.kernel is None:
            raise DeepLearningException(
                'ShadowDenseLayer failed to build kernel.'
            )
        # Build the bias
        if not self.built_bias:
            self.bias = self.add_weight(
                shape=self.units,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype='float32',
                trainable=True,
                name='bias'
            )
            self.built_bias = True
        # Validate the bias
        if self.bias is None:
            raise DeepLearningException(
                'ShadowDenseLayer failed to build bias.'
            )

    def call(self, inputs):
        """
        Call the shadow dense layer.

        See :class:`.Layer` and :meth:`layer.Layer.call`.
        """
        def matmul_add(input):
            x, start = input
            start = tf.squeeze(start)
            output = tf.matmul(x[start+self.offset:], self.kernel)
            if self.use_bias:
                output = tf.add(output, self.bias)
            if self.activation is not None:
                output = self.activation(output)
            return tf.pad(
                output,
                [[start+self.offset, 0], [0, 0]],
                "CONSTANT",
                constant_values=0
            )
        # TODO Remove : Debug section ---
        """start = time.perf_counter()
        output = tf.map_fn(
            fn=matmul_add,
            elems=inputs,
            fn_output_signature=tf.TensorSpec(
                shape=(None, self.units),
                dtype=tf.dtypes.float32
            )
        )
        end = time.perf_counter()
        print(f'{self.name} called in {(1000*(end-start)):.3f} ms')
        return output"""
        # --- TODO Remove : Debug section
        return tf.map_fn(
            fn=matmul_add,
            elems=inputs,
            fn_output_signature=tf.TensorSpec(
                shape=(None, self.units),
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
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(
                self.kernel_initializer
            ),
            'bias_initializer': tf.keras.initializers.serialize(
                self.bias_initializer
            ),
            'kernel_regularizer': tf.keras.regularizers.serialize(
                self.kernel_regularizer
            ),
            'bias_regularizer': tf.keras.regularizers.serialize(
                self.bias_regularizer
            ),
            'activity_regularizer': tf.keras.regularizers.serialize(
                self.activity_regularizer
            ),
            'kernel_constraint': tf.keras.constraints.serialize(
                self.kernel_constraint
            ),
            'bias_constraint': tf.keras.constraints.serialize(
                self.bias_constraint
            )
        })
        # Return updated config
        return config

    @classmethod
    def from_config(cls, config):
        """Use given config data to deserialize the layer"""
        # Instantiate layer
        rdl = cls(**config)
        # Return deserialized layer
        return rdl
