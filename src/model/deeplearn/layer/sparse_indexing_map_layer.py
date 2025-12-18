# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.layer.layer import Layer
import src.main.main_logger as LOGGING
import tensorflow as tf
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class SparseIndexingMapLayer(Layer):
    r"""
    :author: Alberto M. Esmoris Pena

    A sparse indexing map layer handles the building of a
    :class:`tf.MutableHashTable` to govern the indexing of sparse grids.
    More concretely, it enables the computation of :math:`h(k) = v`, a map that
    connects the potentially sparse indices of the active cells on a sparse
    grid (keys, k) to their corresponding sequential indexing (values, v).

    The rationale behind the map :math:`h` is that neighborhoods on sparse grids
    must only consider active cells (other cells don't have useful information).
    To achieve this, the cell-wise features are codified as a non-sparse matrix
    of features :math:`\pmb{F} \in \mathbb{R}^{R \times n_f}` that represents
    the :math:`n_f` features of the :math:`R` active cells. Then, the map
    :math:`h` is used to link the sparse index :math:`k` to its corresponding
    row in :math:`\pmb{F}`, i.e., :math:`v`.

    Note that the explicitly given sequential indices :math:`v` are expected to
    start at one because zero is used to represent the ground vector, i.e., the
    vector of features that represents any non-active cell (typically the
    zero vector).

    :ivar h: The :class:`tf.lookup.experimental.MutableHashTable` used to
        represent the :math:`h(k) = v` map.
    :vartype h: :class:`tf.lookup.experimental.MutableHashTable`.
    :ivar built_h: Whether he mutable hash table has been built or not.
        Initially it is false, but it will be updated once the layer is built.
    :vartype built_h: bool
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, built_h=False, **kwargs):
        """
        See :class:`.Layer` and :meth:`layer.Layer.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Derived attributes
        self.h = None  # Mutable hash table
        self.built_h = built_h  # True if built, False otherwise

    # ---   LAYER METHODS   --- #
    # ------------------------- #
    def build(self, dim_in):
        r"""
        Build the mutable hash table object that will be used to represent the
        :math:`h(k) = v` sparse indexing map.

        See :class:`.Layer` and :meth:`layer.Layer.build`
        """
        # Call parent's build
        super().build(dim_in)
        # Build mutable hash table
        if not self.built_h:
            self.h = tf.lookup.experimental.MutableHashTable(
                tf.int32,
                tf.int32,
                0
            )
        # Validate mutable hash table
        if self.h is None:
            raise DeepLearningException(
                'SparseIndexingMapLayer failed to build mutable hash table.'
            )

    def call(self, inputs):
        r"""
        Remove old values from the mutable hash table and insert the new ones.

        :param inputs: The keys and values of the sparse indexing map
            :math:`h` as two tensors with padding constituting a vectors of keys
            (sparse indices) and a vector of values (sequential indices),
            and the feature space as another tensor.

            Note that inputs[0] gives the tensor with keys, inputs[1] the tensor
            with values, and inputs[2] gives the feature space tensor.

            Note also that the input feature space is given but not modified.
            It is only requested for convenience to define the computational
            graph of the neural network in such a way that the sparse indexing
            map layer is for sure called before a certain point, e.g., before
            computing the first submanifold sparse convolution
            (:class:`.SubmanifoldSpConv3DLayer`) in a sparse convolutional
            neural network (:class:`.SpConv3DPwiseClassif`).

        :return: The given input feature space without modification.
        :rtype: :class:`tf.Tensor`
        """
        # Clear the mutable hash table so the old sparse indexing map is no
        # longer represented by the layer
        self.h.remove(self.h.export()[0])
        # Build the new mutable hash table to represent the current sparse
        # indexing map
        hk, hv = inputs[0], inputs[1]
        mask = hk > -1
        hk = tf.boolean_mask(hk, mask)
        hv = tf.boolean_mask(hv, mask)
        self.h.insert(hk, hv)
        # Return the input feature space as given, for convenience
        return inputs[2]

    # ---   H-MAP METHODS   --- #
    # ------------------------- #
    @tf.function
    def lookup(self, k):
        """
        Look-up for the corresponding sequential indices (v) for given input
        sparse indices (k).

        :param k: The input sparse indices as keys (k).
        :return: The corresponding sequential indices as values (v).
        """
        return self.h.lookup(k)

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def get_config(self):
        """Return necessary data to serialize the layer"""
        # Call parent's config
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        """Use given config data to deserialize the layer"""
        # Instantiate layer
        siml = cls(**config)
        # Return deserialized layer
        return siml

