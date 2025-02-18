# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.layer.layer import Layer
import tensorflow as tf
import time  # TODO Remove : Debug only


# ---   CLASS   --- #
# ----------------- #
class ShadowBatchNormalizationLayer(Layer):
    r"""
    Shadow version of a batch normalization layer, i.e., it works with tensors
    with padding, where the padding is used to represent shadow
    values/points/cells/elements, i.e., those that must not have an impact
    on the computations.

    See :class:`tf.Tensor` and
    :class:`tf.keras.layers.BatchNormalization`.

    A classical batch normalization uses the following four variables
    :math:`\mu_b` (mean of batch), :math:`\sigma^2_b` (variance of batch),
    :math:`\mu_m` (mean for moving average), and :math:`\sigma^2_m` (variance
    for moving average) such that whether :math:`x=\mu` or :math:`x=\sigma^2`
    they can be updated as follows:

    .. math::
        x'_m = (x_m - x_b) M + x_b

    Where :math:`M \in \mathbb{R}` is the momentum for the moving average and
    :math:`x'_m` is the updated value for the variable governing the moving
    average.

    Note that contrary to classical batch normalization where during training
    the :math:`\mu_b, \sigma^2_b` variables are used while for predictions
    the :math:`\mu_m, \sigma^2_m` variables are used, the shadow batch
    normalization layer always considers the batch, i.e., it uses
    :math:`\mu_b` and :math:`\sigma^2_b` during training and for predictions.
    Thus, the final normalization is always computed like:

    .. math::

       \hat{b} = \dfrac{
            \gamma (b - \mu_b)
        }{
            \sqrt{\sigma^2_b + \epsilon}
        }
        + \beta

    Where, :math:`\beta`, :math:`\gamma`, and :math:`\epsilon` are the
    ``beta``, ``gamma``, and ``epsilon`` parameters of the layer. Note that
    :math:`b` represents the values from the batch and :math:`\hat{b}` its
    normalized version.
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
        moving_mean_initializer='zeros',
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
                    self.bn(x[start+self.offset:], training=True),
                    [[start + self.offset, 0]],
                    "CONSTANT",
                    constant_values=0
                )
            else:
                return tf.pad(
                    self.bn(x[start+self.offset:], training=True),
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
            'offset': self.offset,
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
