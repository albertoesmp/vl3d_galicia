# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.layer.layer import Layer
import tensorflow as tf
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class DownsamplingSpConv3DLayer(Layer):
    r"""
    :author: Alberto M. Esmoris Pena

    The :class:`.DownsamplingSpConv3DLayer` behaves like the
    :class:`.SubmanifoldSpConv3DLayer` but with the following changes.

    1. The indices :math:`i` of the active cells are considered for cells from
    the non-downsampled space, i.e., :math:`i \in h^D`.


    2. The :math:`\omega(i)` set is redefined as:

    .. math::

        \omega(i)   = \left\{j_p\right\}_{p=0}^{(w^D)^{n_x}-1}
                    = \left\{j_p\right\}_{p=0}^{(w^D)^3-1}


    3. The :math:`j_p = h(j'_p)` index is computed considering :math:`h` from
    the non-downsampled sparse grid. Also, the :math:`j'_p(i)` function is
    redefined as:

    .. math::

        j'_p(i) = i
            + p \mod w^D
            + \left(
                \left\lfloor\dfrac{p}{w^D}\right\rfloor \mod w^D
            \right)n_2
            + \left(
                \left\lfloor\dfrac{p}{(w^D)^2}\right\rfloor \mod w^D
            \right) n_1 n_2


    :ivar wD: The size (i.e., number of cells) of the downsampling 3D
        convolutional window.
    :vartype w: float
    :ivar f: The number of filters/channels for the convolutions.
    :vartype f: int
    :ivar nf: The number of input features per cell.
    :vartype nf: int
    :ivar ng: The number of output features per cell.
    :vartype ng: int
    :ivar W_initializer: The initializer for the convolutional filters.
    :ivar W_regularizer: The regularizer for the convolutional filters.
    :ivar W_constraint: The constraint for the convolutional filters.
    :ivar siml: The sparse indexing map layer that is needed to translate
        sparse indices to sequential indices (i.e., :math:`h(k) = v` map).
    :vartype siml: :class:`.SparseIndexingMapLayer`.
    :ivar wDsq: :math:`(w^D)^2`
    :vartype wDsq: int
    :ivar nwD: :math:`(w^D)^{n_x} = (w^D)^{3}`
    :vartype nw: int
    :ivar W: The tensor of weights
        :math:`\mathcal{W} \in \mathbb{R}^{f \times n_f \times n_g}`
        such that its slices are the matrices
        representing the trainable parameters for each convolutional filter.
    :vartype W: :class:`tf.Tensor`
    :ivar built_W: Whether the :math:`\mathcal{W}` tensor of weights has been
        built or not. Initially it is false, but it will be updated once the
        layer is built.
    :vartype built_W: bool
    :ivar p: The sequential indexing vector representing the upsampling
        convolutional windows.
    :vartype p: :class:`tf.Tensor` of :class:`tf.int32`
    :ivar built_p: Whether the indexing vector representing the upsampling
        convolutional windows has been built or not. Initially it is false, but
        it will be updated once the layer is built.
    :vartype built_p: bool
    :ivar pon1: Cache for :math:`p \mod w^D`.
    :vartype pon1: :class:`tf.Tensor` of :class:`tf.int32`
    :ivar built_pon1: Whether the cache for :math:`p \mod w^D` has been
        built or not. Initially it is false, but it will be updated once the
        layer is built.
    :vartype built_pon1: bool
    :ivar ponw: Cache for
        :math:`\left(\lfloor\dfrac{p}{w^D}\rfloor \mod w^D\right)`.
    :vartype ponw: :class:`tf.Tensor` of :class:`tf.int32`
    :ivar built_ponw: Whether the cache for
        :math:`\left(\lfloor\dfrac{p}{w^D}\rfloor \mod w^D\right)`
        has been built or not. Initially it is false, but it will be updated
        once the layer is built.
    :vartype built_ponw: bool
    :ivar ponw2: Cache for
        :math:`\left(\lfloor\dfrac{p}{(w^D)^2}\rfloor \mod w^D\right)`.
    :vartype ponw2: :class:`tf.Tensor` of :class:`tf.int32`
    :ivar built_ponw2: Whether the cache for
        :math:`\left(\lfloor\dfrac{p}{(w^D)^2}\rfloor \mod w^D\right)`
        has been built or not. Initially it is false, but it will be updated
        once the layer is built.
    :vartype built_ponw2: bool
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(
        self,
        wD,
        f,
        nf,
        ng,
        built_W=False,
        W_initializer = None,
        W_regularizer = None,
        W_constraint = None,
        siml=None,
        built_p=False,
        built_pon1=False,
        built_ponw=False,
        built_ponw2=False,
        **kwargs
    ):
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.wD = wD  # Size of convolutional window
        self.f = f  # Number of convolutional filters (channels)
        self.nf = nf  # Dimensionality of input feature space
        self.ng = ng  # Dimensionality of output feature space
        self.W_initializer = tf.keras.initializers.get(W_initializer)
        self.W_regularizer = tf.keras.regularizers.get(W_regularizer)
        self.W_constraint = tf.keras.constraints.get(W_constraint)
        self.siml = siml  # Layer handling the h map
        # Validate attributes
        if self.wD is None or self.wD < 1:
            raise DeepLearningException(
                'DownsamplingSpConv3DLayer cannot be instantiated without '
                'a convolutional window.'
            )
        if self.f is None or self.f < 1:
            raise DeepLearningException(
                'DownsamplingSpConv3DLayer cannot be instantiated if the '
                'number of convolutional filters or channels is not strictly '
                'greater than zero.'
            )
        if self.nf is None or self.nf < 1:
            raise DeepLearningException(
                'DownsamplingSpConv3DLayer cannot be instantiated if the '
                'dimensionality of the input feature space is not strictly '
                'greater than zero.'
            )
        if self.ng is None or self.ng < 1:
            raise DeepLearningException(
                'DownsamplingSpConv3DLayer cannot be instantiated if the '
                'dimensionality of the output feature space is not strictly '
                'greater than zero.'
            )
        # Derived attributes
        self.wDsq = self.wD * self.wD  # Squared of cells/axis of conv. win.
        self.nwD = self.wDsq * self.wD  # Cells per convolutional window
        # Attributes initialized to None (derived when building)
        self.W = None  # Tensor of convolutional weights
        self.built_W = built_W  # True if built, False otherwise
        self.p = None  # Downsampling sequential indexing vector
        self.built_p = built_p  #  True if built, False otherwise
        self.pon1 = None  # Cache: p mod w^D
        self.built_pon1 = built_pon1  # True if built, False otherwise
        self.ponw = None  # Cache: p/w^D mod w^D
        self.built_ponw = built_ponw  #  True if built, False otherwise
        self.ponw2 = None  # Cache: p/(w^D)^2 mod w^D
        self.built_ponw2 = built_ponw2  # True if built, False otherwise

    # ---   LAYER METHODS   --- #
    # ------------------------- #
    def build(self, dim_in):
        r"""
        Build the :math:`\mathcal{W} \in \mathbb{R}^{f \times n_f \times n_g}`
        tensor representing the weights for each of the :math:`f` convolutional
        filters. Also builds the caches.

        See :class:`.Layer` and :meth:`layer.Layer.build`.
        """
        # Call parent's build
        super().build(dim_in)
        # Build the convolutional weights
        if not self.built_W:
            self.W = self.add_weight(
                shape=(self.f, self.nf, self.ng),
                initializer=self.W_initializer,
                regularizer=self.W_regularizer,
                constraint=self.W_constraint,
                dtype='float32',
                trainable=True,
                name='W'
            )
            self.built_W = True
        # Validate the convolutional weights
        if self.W is None:
            raise DeepLearningException(
                'DownsamplingSpConv3DLayer failed to build convolutional '
                'weights.'
            )
        # Build sequential indexing vector representing any convolutional window
        if not self.built_p:
            self.p = tf.constant(
                np.arange(self.nwD, dtype=np.int32),
                dtype=tf.int32
            )
        # Validate the sequential indexing vector for downsampling convolutions
        if self.p is None:
            raise DeepLearningException(
                'DownsamplingSpConv3DLayer failed to build sequential indexing '
                'vector representing downsampling convolutional windows.'
            )
        # Build cached (p mod (2w+1))
        if not self.built_pon1:
            self.pon1 = tf.math.floormod(self.p, self.wD)
        # Validate cached (p mod w)
        if self.pon1 is None:
            raise DeepLearningException(
                'DownsamplingSpConv3DLayer failed to build pon1 cache.'
            )
        # Build cached (p/(2w+1) mod (2w+1))
        if not self.built_ponw:
            self.ponw = tf.cast(
                tf.math.floormod(tf.floor(self.p / self.wD), self.wD),
                dtype=tf.int32
            )
        # Validate cached (p/(2w+1) mod (2w+1))
        if self.ponw is None:
            raise DeepLearningException(
                'DownsamplingSpConv3DLayer failed to build ponw cache.'
            )
        # Build cached (p/(2w+1)^2 mod (2w+1))
        if not self.built_ponw2:
            self.ponw2 = tf.cast(
                tf.math.floormod(tf.floor(self.p / self.wDsq), self.wD),
                dtype=tf.int32
            )
        # Validate cached (p/(2w+1)^2 mod (2w+1))
        if self.ponw2 is None:
            raise DeepLearningException(
                'DownsamplingSpConv3DLayer failed to build ponw2 cache.'
            )

    def call(self, inputs, training=False, mask=False):
        r"""
        Compute the downsampling convolutions.

        See :class:`.DownsamplingSpConv3DLayer` for the maths.

        :param inputs: The feature space matrices as a 3D tensor with padding,
            the :math:`h^D` vector of indices that gives the indices
            of the active cells in the non-downsampled sparse grid that are
            the centers for the downsampling convolutional windows, the
            matrix :math:`\pmb{N} \in \mathbb{Z}^{n_x \times K}` of
            axis-wise partitions in the non-downsampled space (where :math:`K`
            is the batch size), and the start row indices.

            Note that inputs[0] gives the feature spaces,
            inputs[1] the vector of active cell indices to center the
            downsampling convolutional window in the non-downsampled sparse
            grid, inputs[2] the axis-wise partitions, inputs[3] the start
            row-index for the downsampled space, and inputs[4] the start
            row-index for the non-downsampled space (source, src). The start
            row-index allows to handle the padding.

        :return: The output feature space matrices as a tensor with padding.
        :rtype: :class:`tf.Tensor`
        """
        def convolve(input):
            # Extract inputs
            F, hD, n, start, src_start = input
            start = tf.squeeze(start)
            src_start = tf.squeeze(src_start)
            # Gather relevant values for computations
            F = tf.gather(F, tf.range(src_start, tf.shape(F)[0]))
            hD = tf.gather(hD, tf.range(start, tf.shape(hD)[0]))
            # Compute convolution
            ny, nz = n[1], n[2]
            nynz = ny*nz
            omega = tf.transpose(
                hD + tf.expand_dims(
                    self.pon1 + self.ponw * nz + self.ponw2 * nynz,
                    axis=1
                )
            )
            G = tf.einsum(
                'ijk,lkm->im',
                tf.gather(
                    F,
                    tf.ensure_shape(self.siml.lookup(omega), omega.shape)
                ),
                self.W
            )
            # Add ground vector and padding (zeros) at the beginning of G
            return tf.pad(
                G,
                [[1+start, 0], [0, 0]],
                "CONSTANT",
                constant_values=0
            )
        # Convolve each element in the batch
        # TODO Remove : Debug section ---
        """start = time.perf_counter()
        output = tf.map_fn(
            convolve,
            inputs,
            fn_output_signature=tf.TensorSpec(
                shape=(None, self.ng),
                dtype=tf.float32
            )
        )
        end = time.perf_counter()
        print(f'{self.name} called in {(1000*(end-start)):.3f} ms')
        return output"""
        # --- TODO Remove : Debug section
        return tf.map_fn(
            convolve,
            inputs,
            fn_output_signature=tf.TensorSpec(
                shape=(None, self.ng),
                dtype=tf.float32
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
            'wD': self.wD,
            'f': self.f,
            'nf': self.nf,
            'ng': self.ng,
            'W_initializer': tf.keras.initializers.serialize(
                self.W_initializer
            ),
            'W_regularizer': tf.keras.regularizers.serialize(
                self.W_regularizer
            ),
            'W_constraint': tf.keras.constraints.serialize(self.W_constraint),
            'siml': self.siml
        })
        # Return updated config
        return config

    @classmethod
    def from_config(cls, config):
        """Use given config data to deserialize the layer"""
        # Instantiate layer
        dsc3Dl = cls(**config)
        # Return deserialized layer
        return dsc3Dl
