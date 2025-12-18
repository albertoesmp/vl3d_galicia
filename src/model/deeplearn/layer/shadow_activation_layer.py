# ---   IMPORTS   --- #
# ⨪------------------ #
from src.model.deeplearn.layer.layer import Layer
import tensorflow as tf
import time  # TODO Remove : Debug only


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
            # TODO Rethink : Should use offset=1 for SpConv3DPwiseClassif
            #print(f'{self.name} F[:3]:\n{x[start+self.offset:][:3]}')  # TODO Remove : Debug
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
        # TODO Remove : Debug section ---
        """start = time.perf_counter()
        output = tf.map_fn(
            activate,
            inputs,
            fn_output_signature=tf.TensorSpec(
                shape=[None for k in range(1, len(tf.shape(inputs[0])))],
                dtype=tf.dtypes.float32
            )
        )
        end = time.perf_counter()
        print(f'{self.name} called in {(1000*(end-start)):.3f} ms')
        return output"""
        # --- TODO Remove : Debug section
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
            'offset': self.offset,
            'act': self.act
        })
        # Return updated config
        return config

    @classmethod
    def from_config(cls, config):
        """Use given config data to deserialize the layer"""
        # Instantiate layer
        act = config['act']
        config['act'] = None
        act = cls(act, **config)
        # Return deserialized layer
        return act
