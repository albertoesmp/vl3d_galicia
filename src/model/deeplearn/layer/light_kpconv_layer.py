# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.layer.kpconv_layer import KPConvLayer
from src.report.light_kpconv_layer_report import LightKPConvLayerReport
from src.plot.light_kpconv_layer_plot import LightKPConvLayerPlot
import src.main.main_logger as LOGGING
import tensorflow as tf
import numpy as np
import time

# ---   CLASS   --- #
# ----------------- #
class LightKPConvLayer(KPConvLayer):
    r"""
    :author: Alberto M. Esmoris Pena

    A light kernel point convolution layer redefines the typical
    :class:`.KPConvLayer` to work with a single matrix of weights (instead of
    one per kernel point) and supporting an optional scale factor for each
    kernel point and input feature pair. The new tensor of weights is thus a
    matrix
    :math:`\pmb{W} \in \mathbb{R}^{D_{\mathrm{in}} \times D_{\mathrm{out}}}`.
    Also, all the scale factors can be represented as a matrix
    :math:`\pmb{A} \in \mathbb{R}^{m_q \times D_{\mathrm{in}}}`.

    See :class:`.KPConvLayer`.

    :ivar A_trainable: Whether the scale factors are trainable (i.e., can be
        updated through backpropagation) or not.
    :vartype A_trainable: bool
    :ivar built_A: Whether the matrix of scale factors
        :math:`\pmb{A} \in \mathbb{R}^{m_q \times D_{\mathrm{in}}}` is built.
        Initially it is false, but it will be updated once the layer is built.
    :vartype built_A: bool
    :ivar A_initializer: The initializer for the scale factors matrix.
    :ivar A_regularizer: The regularizer for the scale factors matrix.
    :ivar A_constraint: The constraint for the scale factors matrix.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(
        self,
        A_trainable=True,
        built_A=False,
        A_initializer='ones',
        A_regularizer=None,
        A_constraint=None,
        **kwargs
    ):
        """
        See :class:`.KPConvLayer` and :meth:`.KPConvLayer.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.A_trainable = A_trainable
        self.A_initializer = tf.keras.initializers.get(A_initializer)
        self.A_regularizer = tf.keras.regularizers.get(A_regularizer)
        self.A_constraint = tf.keras.constraints.get(A_constraint)
        # Attributes initialized to None (derived when building)
        self.A = None  # The scale factors for pairs (kpoint, infeature)
        self.built_A = built_A  # True means built, False means not built

    # ---   LAYER METHODS   --- #
    # ------------------------- #
    def build(self, dim_in):
        r"""
        Build the whole Light KPConv layer by calling the parent's build method
        (see :meth:`.KPConvLayer.build`) with the derived
        :meth:`.LightKPConvLayer.build_W` logic and also build the
        :math:`\pmb{A} \in \mathbb{R}^{m_q \times D_{\mathrm{in}}}` matrix.

        See :class:`.KPConvLayer` and :meth:`.KPConvLayer.build`).
        """
        # Call parent's build
        super().build(dim_in)
        # Find the dimensionality of the input feature space
        Din = dim_in[-2][-1]
        # Build the matrix of scale factors
        if not self.built_A:
            self.A = self.add_weight(
                shape=(self.num_kernel_points, Din),
                initializer=self.A_initializer,
                regularizer=self.A_regularizer,
                constraint=self.A_constraint,
                dtype='float32',
                trainable=self.A_trainable,
                name='A'
            )
            self.built_A = True

    def compute_output_features(self, X, NX, NF):
        r"""
        Assist the :meth:`.KPConvLayer.call` method by applying the point-wise
        convolutions with the single matrix of weights
        :math:`\pmb{W} \in \mathbb{R}^{D_{\mathrm{in}} \times D_{\mathrm{out}}}`
        and the scale factors in
        :math:`\pmb{A} \in \mathbb{R}^{m_q \times D_{\mathrm{in}}}`.

        The new operation consists of redefining the kernel point convolution
        to work with the light kernel (where
        :math:`\mathcal{Q} = (\pmb{Q}, \pmb{W}))` instead of
        :math:`\mathcal{Q} = (\pmb{Q}, \mathcal{W}))`)
        such that:

        .. math::
            \begin{aligned}
            \left(\pmb{P} * \mathcal{Q} \right) &=\;
                \sum_{\pmb{x}_{j*} \in \mathcal{N}_{\pmb{x}_{i*}}} \left[
                    \sum_{k=1}^{m_q}{
                        \max \; \left\{
                            0,\,
                            1 - \dfrac{
                                \lVert{
                                    \pmb{x}_{j*} -
                                    \pmb{x}_{i*} -
                                    \pmb{q}_{k*}
                                }\rVert
                            }{
                                \sigma
                            }
                        \right\} \biggl(
                            \operatorname{diag}\left(\pmb{a}_{k*}\right)
                            \pmb{W}
                        \biggr)^{\intercal}
                    }
            \right] \pmb{f}_{j*} \\  &=\;
            \sum_{\pmb{x}_{j*} \in \mathcal{N}_{\pmb{x}_{i*}}}
                \left(\operatorname{diag}\left[\sum_{k=1}^{m_q}{
                    \max \; \left\{
                        0,
                        1 - \dfrac{
                            \lVert
                                \pmb{x}_{j*} -
                                \pmb{x}_{i*} -
                                \pmb{q}_{k*}
                            \rVert
                        }{
                            \sigma
                        }
                    \right\}
                    \pmb{a}_{k*}
                }
            \right] \pmb{W}\right)^{\intercal} \pmb{f}_{j*}
            \end{aligned}

        See :meth:`.KPConvLayer.compute_output_features` and
        :meth:`.KPConvLayer.call` for further details.
        """
        # Compute linear correlations (K x R x kappa x m_q)
        Wc = NX-tf.expand_dims(X, axis=2)  # xj - xi
        Wc = tf.tile(  # xj - xi - qk
            tf.expand_dims(Wc, axis=3),
            [1, 1, 1, self.num_kernel_points, 1]
        ) - self.Q
        Wc = 1-tf.sqrt(tf.reduce_sum(tf.square(Wc), axis=4))/self.sigma
        Wc = tf.maximum(0., Wc)
        # Compute output features with light kernel point convolutions
        Wc = tf.matmul(Wc, self.A)  # Scaled correlations
        return tf.matmul(tf.reduce_sum(Wc * NF, axis=2), self.W)  # Output

    # ---   BUILDING METHODS   --- #
    # ---------------------------- #
    def build_W(self, Din):
        r"""
        Assist the :meth:`.LightKPConvLayer.build` method in building the
        :math:`\mathcal{W} \in \mathbb{R}^{D_{\mathrm{in}} \times D_{\mathrm{out}}}`
        tensor, i.e., the convolution weights.

        See :class:`.KPConvLayer` and :meth:`.KPConvLayer.build`.

        :param Din: The dimensionality of the input feature space
            :math:`D_{\mathrm{in}}`.
        :type Din: int
        """
        return self.add_weight(
            shape=(Din, self.Dout),
            initializer=self.W_initializer,
            regularizer=self.W_regularizer,
            constraint=self.W_constraint,
            dtype='float32',
            trainable=True,
            name='W'
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
            'A_trainable': self.A_trainable,
            'A_initializer': tf.keras.initializers.serialize(
                self.A_initializer
            ),
            'A_regularizer': tf.keras.regularizers.serialize(
                self.A_regularizer
            ),
            'A_constraint': tf.keras.constraints.serialize(
                self.A_constraint
            )
        })
        # Return updated config
        return config

    @classmethod
    def from_config(cls, config):
        """Use given config data to deserialize the layer"""
        # Instantiate layer
        lkpcl = cls(**config)
        # Return deserialized layer
        return lkpcl

    # ---   PLOTS and REPORTS   --- #
    # ----------------------------- #
    def export_representation(
        self, dir_path, out_prefix=None, Wpast=None, Apast=None
    ):
        """
        Export a set of files representing the state of the kernel for both
        the structure (Q) and the weights (W).

        :param dir_path: The directory where the representation files will be
            exported.
        :type dir_path: str
        :param out_prefix: The output prefix to name the output files.
        :type out_prefix: str
        :param Wpast: The weights of the kernel in the past.
        :type Wpast: :class:`np.ndarray` or :class:`tf.Tensor` or None
        :param Apast: The scale factors in the past.
        :type Apast: :class:`np.ndarray` or :class:`tf.Tensor` or None
        :return: Nothing at all, but the representation is exported as a set
            of files inside the given directory.
        """
        # Check dir_path has been given
        if dir_path is None:
            LOGGING.LOGGER.debug(
                'LightKPConvLayer.export_representation received no '
                'dir_path.'
            )
            return
        # Export the values (report) and the plots
        LOGGING.LOGGER.debug(
            'Exporting representation of light KPConv layer to '
            f'"{dir_path}" ...'
        )
        # Export report
        start = time.perf_counter()
        LightKPConvLayerReport(
            np.array(self.Q), np.array(self.W), np.array(self.A)
        ).to_file(dir_path, out_prefix=out_prefix)
        # Export plots
        LightKPConvLayerPlot(
            Q=np.array(self.Q),
            W=np.array(self.W),
            A=np.array(self.A),
            Wpast=np.array(Wpast) if Wpast is not None else None,
            Apast=np.array(Apast) if Apast is not None else None,
            sigma=self.sigma,
            name=self.name,
            path=dir_path
        ).plot(out_prefix=out_prefix)
        # Log time
        end = time.perf_counter()
        LOGGING.LOGGER.debug(
            'Representation of light KPConv layer exported to '
            f'"{dir_path}" in {end-start:.3f} seconds.'
        )
