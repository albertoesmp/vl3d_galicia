# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.layer.light_kpconv_layer import LightKPConvLayer

# ---   CLASS   --- #
# ----------------- #
class StridedLightKPConvLayer(LightKPConvLayer):
    r"""
    :author: Alberto M. Esmoris Pena

    Strided version of the :class:`.LightKPConvLayer` layer. Instead of
    transforming :math:`R` input points with :math:`D_{\mathrm{in}}` features
    into :math:`R` output points with :math:`D_{\mathrm{out}}` features, it
    transforms :math:`R_1` input points with :math:`D_{\mathrm{in}}` features
    into :math:`R_2` output points with :math:`D_{\mathrm{out}}` features,
    where typically :math:`R_1 > R_2`.

    See :class:`.LightKPConvLayer` and :class:`.StridedKPConvLayer`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        See :class:`.Layer` and :meth:`.Layer.__init__`.
        Also, see :class:`.LightKPConvLayer` and
        :meth:`.LightKPConvLayer.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)

    # ---   LAYER METHODS   --- #
    # ------------------------- #
    def build(self, dim_in):
        """
        See :class:`.LightKPConvLayer` and :meth:`.LightKPConvLayer.build`.
        """
        # Call parent's build
        super().build(dim_in)

    def call(self, inputs, training=False, mask=False):
        r"""
        Compute the strided version of light KPConv. The main difference with
        respect to the :class:`.LightKPConvLayer` layer is that the
        :math:`\kappa` neighbor points
        :math:`\pmb{x}_{j*} \in \mathcal{N}^{D}_{\pmb{x}_{i*}}` are now defined
        such that :math:`\pmb{x}_{i*}` is a point from a downsampled structure
        space :math:`\pmb{X_{b}} \in \mathbb{R}^{R_2 \times n_x}`, while
        the :math:`\pmb{x}_{j*}` points belong to the structure space before
        downsampling :math:`\pmb{X_a} \in \mathbb{R}^{R_1 \times n_x}`.

        See :class:`.LightKPConvLayer`, :class:`.StridedKPConvLayer`, and
        :meth:`.StridedKPConvLayer.call`.

        :param inputs: The input such that:

            -- inputs[0]
                is the structure space tensor representing the geometry of the
                many receptive fields in the batch.

                .. math::
                    \mathcal{X_a} \in \mathbb{R}^{K \times R_1 \times n_x}

            -- inputs[1]
                is the structure space tensor representing the geometry of the
                many receptive fields in the batch.

                .. math::
                    \mathcal{X_b} \in \mathbb{R}^{K \times R_2 \times n_x}

            -- inputs[2]
                is the feature space tensor representing the features of the
                many receptive fields in the batch.

                .. math::
                    \mathcal{F} \in \mathbb{R}^{K \times R_1 \times n_f}

            -- inputs[3]
                is the indexing tensor representing the neighborhoods of
                :math:`\kappa` neighbors in the non downsampled space for each
                point in the downsampled.

                .. math::
                    \mathcal{N}^D \in \mathbb{Z}^{K \times R_2 \times \kappa}

        :return: The output feature space
            :math:`\mathcal{Y} \in \mathbb{R}^{K \times R_2 \times D_{\mathrm{out}}}`.
        """
        # Extract input
        Xa = inputs[0]
        Xb = inputs[1]
        Fa = inputs[2]
        ND = inputs[3]
        # Gather neighborhoods (K x Rb x kappa x n_x=3 or n_d)
        NX, NF = self.gather_neighborhoods(ND, Xa, Fa)
        # Return output features
        return self.compute_output_features(Xb, NX, NF)
