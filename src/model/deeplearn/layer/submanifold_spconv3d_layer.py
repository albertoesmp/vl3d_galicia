# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.layer.layer import Layer
import tensorflow as tf
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class SubmanifoldSpConv3DLayer(Layer):
    r"""
    :author: Alberto M. Esmoris Pena

    A submanifold sparse 3D convolution layer consists of a dense convolution
    that is applied on a 3D sparse grid centering the window only on the active
    cells. The active cells are those cells that contain at least one point
    from the original input (typically, the 3D input point cloud).

    A submanifold sparse 3D convolution layer is defined by a map
    :math:`h : X \subset \mathbb{Z}_{\geq 0} \rightarrow Y \subset \mathbb{Z}_{\geq 0}`.
    Note that both :math:`X` and :math:`Y` are **finite** subsets of the
    non-negative integers. Moreover, when the set :math:`Y` is non-empty (i.e.,
    there is at least one active cell) it satisfies the following:

    1. :math:`0 \in Y`

    2. :math:`\forall y \in Y,\, y > 0, (y-1) \in Y`

    In other words, :math:`h` maps the indices of sparse active cells to a
    finite set of sequential indices. These indices are used to link each
    active cell to its corresponding row in the matrix of features
    :math:`\pmb{F} \in \mathbb{R}^{(1+\lvert{X}\rvert) \times n_f}`, where
    :math:`n_f` is the dimensionality of the input feature space and the
    :math:`1` in :math:`(1+\lvert{X}\rvert)` accounts for the ground value
    (i.e., :math:`0`) represented with the first row of the matrix
    :math:`\pmb{F}`.

    Thus, the submanifold 3D convolution centered on the :math:`i`-th cell of
    a sparse 3D grid with a window size :math:`w \in \mathbb{Z}_{>0}` implies
    computing (note that for submanifold sparse convolutions :math:`w` is a
    half size instead of a full size, thus it must be symmetrically applied
    to both directions of each axis):

    .. math::

        \omega(i)   = \left\{j_p\right\}_{p=0}^{(2w+1)^{n_x}-1}
                    = \left\{j_p\right\}_{p=0}^{(2w+1)^3-1}

    Where :math:`j_p` is the index of a row in the matrix of input features
    :math:`\pmb{F}` and :math:`n_x` is the dimensionality of the structure
    space (for the case of 3D convolutions :math:`n_x = 3`).

    More concretely, :math:`j_p = h(j'_p)` with:

    .. math::

        j'_p(i) = i
            - w (1 + n_3 + n_3 n_2)
            + p \mod (2w+1)
            + \left(
                \left\lfloor\dfrac{p}{2w+1}\right\rfloor \mod (2w+1)
            \right)n_2
            + \left(
                \left\lfloor\dfrac{p}{(2w+1)^2}\right\rfloor \mod (2w+1)
            \right) n_1 n_2

    Where :math:`n_1` is the number of partitions along the :math:`y` axis and
    :math:`n_2` is the number of partitions along the :math:`z` axis. Note that
    these values come from the vector of axis-wise partitions
    :math:`\pmb{n} = (n_1, n_2, n_3)`.

    Finally, the output matrix of features
    :math:`\pmb{G} \in \mathbb{R}^{(1 + \lvert{X}\rvert) \times n_g}`
    will be computed as follows:

    .. math::

        \pmb{g}_{h(i)*} = \sum_{j \in \omega(i)} \sum_{l=1}^{f}{
            \pmb{f}_{j*} \pmb{W}_{l}
        }

    Where :math:`n_g \in \mathbb{Z}_{>0}` is the number of output features,
    :math:`f \in \mathbb{Z}_{>0}` is the number of convolutional filters (or
    channels) and, for convenience, the first row of :math:`\pmb{G}` is
    :math:`\pmb{g}_{0*} = \pmb{0}` (so it represents the ground values, i.e.,
    zeroes). Note also that :math:`\pmb{f}_{j*}` refers to the :math:`j`-th
    row if the input feature space (:math:`\pmb{F}`) and :math:`\pmb{W}_l`
    refers to the matrix of weights that represents the :math:`l`-th
    convolutional filter.

    :ivar w: The half size (i.e., number of cells along each direction of each
        axis) of the submanifold 3D convolutional window.
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
    :ivar wp: The size (not half size) of the submanifold 3D convolutional
        window in terms of number of cells, i.e., :math:`2w+1`.
    :vartype wp: int
    :ivar wpsq: :math:`(2w+1)^2`
    :vartype wpsq: int
    :ivar nw: :math:`(2w+1)^{n_x} = (2w+1)^{3}`
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
    :ivar p: The sequential indexing vector representing the submanifold
        convolutional windows.
    :vartype p: :class:`tf.Tensor` of :class:`tf.int32`
    :ivar built_p: Whether the indexing vector representing the submanifold
        convolutional windows has been built or not. Initially it is false, but
        it will be updated once the layer is built.
    :vartype built_p: bool
    :ivar pon1: Cache for :math:`p \mod (2w+1)`.
    :vartype pon1: :class:`tf.Tensor` of :class:`tf.int32`
    :ivar built_pon1: Whether the cache for :math:`p \mod (2w+1)` has been
        built or not. Initially it is false, but it will be updated once the
        layer is built.
    :vartype built_pon1: bool
    :ivar ponw: Cache for
        :math:`\left(\left\lfloor\dfrac{p}{2w+1}\right\rfloor \mod (2w+1)\right)`.
    :vartype ponw: :class:`tf.Tensor` of :class:`tf.int32`
    :ivar built_ponw: Whether the cache for
        :math:`\left(\left\lfloor\dfrac{p}{2w+1}\right\rfloor \mod (2w+1)\right)`
        has been built or not. Initially it is false, but it will be updated
        once the layer is built.
    :vartype built_ponw: bool
    :ivar ponw2: Cache for
        :math:`\left(\left\lfloor\dfrac{p}{(2w+1)^2}\right\rfloor \mod (2w+1)\right)`.
    :vartype ponw2: :class:`tf.Tensor` of :class:`tf.int32`
    :ivar built_ponw2: Whether the cache for
        :math:`\left(\left\lfloor\dfrac{p}{(2w+1)^2}\right\rfloor \mod (2w+1)\right)`
        has been built or not. Initially it is false, but it will be updated
        once the layer is built.
    :vartype built_ponw2: bool
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(
            self,
            w,
            f,
            nf,
            ng,
            built_W=False,
            W_initializer=None,
            W_regularizer=None,
            W_constraint=None,
            siml=None,
            built_p=False,
            built_pon1=False,
            built_ponw=False,
            built_ponw2=False,
            **kwargs
    ):
        """
        See :class:`.Layer` and :meth:`layer.Layer.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.w = w  # Size (half) of convolutional window
        self.f = f  # Number of convolutional filters (channels)
        self.nf = nf  # Dimensionality of input feature space
        self.ng = ng  # Dimensionality of output feature space
        self.W_initializer = tf.keras.initializers.get(W_initializer)
        self.W_regularizer = tf.keras.regularizers.get(W_regularizer)
        self.W_constraint = tf.keras.constraints.get(W_constraint)
        self.siml = siml  # Layer handling the h map
        # Validate attributes
        if self.w is None or self.w < 1:
            raise DeepLearningException(
                'SubmanifoldSpConv3DLayer cannot be instantiated without '
                'a convolutional window.'
            )
        if self.f is None or self.f < 1:
            raise DeepLearningException(
                'SubmanifoldSpConv3DLayer cannot be instantiated if the '
                'number of convolutional filters or channels is not strictly '
                'greater than zero.'
            )
        if self.nf is None or self.nf < 1:
            raise DeepLearningException(
                'SubmanifoldSpConv3DLayer cannot be instantiated if the '
                'dimensionality of the input feature space is not strictly '
                'greater than zero.'
            )
        if self.ng is None or self.ng < 1:
            raise DeepLearningException(
                'SubmanifoldSpConv3DLayer cannot be instantiated if the '
                'dimensionality of the output feature space is not strictly '
                'greater than zero.'
            )
        # Derived attributes
        self.wp = 2*self.w+1  # Number of cells per axis of conv. window
        self.wpsq = self.wp * self.wp  # Squared of cells per axis of conv. win.
        self.nw = self.wpsq * self.wp  # Number cells per convolutional window
        # Attributes initialized to None (derived when building)
        self.W = None  # Tensor of convolutional weights
        self.built_W = built_W  # True if built, False otherwise
        self.p = None  # Submanifold sequential indexing vector
        self.built_p = built_p  #  True if built, False otherwise
        self.pon1 = None  # Cache: p mod (2w+1)
        self.built_pon1 = built_pon1  # True if built, False otherwise
        self.ponw = None  # Cache: p/(2w+1) mod (2w+1)
        self.built_ponw = built_ponw  #  True if built, False otherwise
        self.ponw2 = None  # Cache: p/(2w+1)^2 mod (2w+1)
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
                'SubmanifoldSpConv3DLayer failed to build convolutional '
                'weights.'
            )
        # Build sequential indexing vector representing any convolutional window
        if not self.built_p:
            self.p = tf.constant(
                np.arange(self.nw, dtype=np.int32),
                dtype=tf.int32
            )
        # Validate the sequential indexing vector for submanifold convolutions
        if self.p is None:
            raise DeepLearningException(
                'SubmanifoldSpConv3DLayer failed to build sequential indexing '
                'vector representing submanifold convolutional windows.'
            )
        # Build cached (p mod (2w+1))
        if not self.built_pon1:
            self.pon1 = tf.math.floormod(self.p, self.wp)
        # Validate cached (p mod w)
        if self.pon1 is None:
            raise DeepLearningException(
                'SubmanifoldSpConv3DLayer failed to build pon1 cache.'
            )
        # Build cached (p/(2w+1) mod (2w+1))
        if not self.built_ponw:
            self.ponw = tf.cast(
                tf.math.floormod(tf.floor(self.p / self.wp), self.wp),
                dtype=tf.int32
            )
        # Validate cached (p/(2w+1) mod (2w+1))
        if self.ponw is None:
            raise DeepLearningException(
                'SubmanifoldSpConv3DLayer failed to build ponw cache.'
            )
        # Build cached (p/(2w+1)^2 mod (2w+1))
        if not self.built_ponw2:
            self.ponw2 = tf.cast(
                tf.math.floormod(tf.floor(self.p / self.wpsq), self.wp),
                dtype=tf.int32
            )
        # Validate cached (p/(2w+1)^2 mod (2w+1))
        if self.ponw2 is None:
            raise DeepLearningException(
                'SubmanifoldSpConv3DLayer failed to build ponw2 cache.'
            )

    def call(self, inputs, training=False, mask=False):
        r"""
        Compute the submaniold convolutions.

        See :class:`.SubmanifoldSpConv3DLayer` for the maths.

        :param inputs: The feature space matrices as a 3D tensor with padding,
            the submanifold map :math:`h` as a tensor of keys (k) with padding
            the matrix :math:`\pmb{N} \in \mathbb{Z}^{n_x \times K}` of
            axis-wise partitions (where :math:`K` is the batch size), and
            the start row indices.

            Note that inputs[0] gives the feature spaces, inputs[1] the keys
            of the map h, inputs[2] the axis-wise
            partitions, and inputs[3] the start row-index for the elements
            in the batch so the padding can be ignored.

        :return: The output feature space matrices as a tensor.
        :rtype: :class:`tf.Tensor`
        """
        def convolve(input):
            # Extract inputs
            F, hk, n, start = input
            start = tf.squeeze(start)
            # Gather relevant values for computations
            F = tf.gather(F, tf.range(start, tf.shape(F)[0]))
            hk = tf.gather(hk, tf.range(start, tf.shape(hk)[0]))
            # Compute convolution
            ny, nz = n[1], n[2]
            nynz = ny*nz
            ishift = self.w*(1+nz+nynz)
            omega = tf.transpose(
                hk + tf.expand_dims(
                    self.pon1 + self.ponw * nz + self.ponw2 * nynz - ishift,
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
        # Update config with custom attributes
        config.update({
            # Base attributes
            'w': self.w,
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
        ssc3Dl = cls(**config)
        # Return deserialized layer
        return ssc3Dl