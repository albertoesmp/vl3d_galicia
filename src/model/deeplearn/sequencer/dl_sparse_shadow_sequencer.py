# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.sequencer.dl_abstract_sequencer import \
    DLAbstractSequencer
from src.model.deeplearn.deep_learning_exception import DeepLearningException
import tensorflow as tf
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class DLSparseShadowSequencer(DLAbstractSequencer):
    """
    :author: Alberto M. Esmoris Pena

    A deep learning sequencer that governs how the input data is fed to a
    neural network. It handles two main points:

    1) **Memory management**: Explicitly take control of the memory handling
        logic to prevent undesired scenarios or memory exhaustion due to
        TensorFlow loading the full array of batches into the GPU memory.

    2) **Shadow tensors**: The data for models based on sparse grids
        (e.g., :class:`.SpConv3DPwiseClassif`) must be converted from ragged
        tensors to shadow tensors. Ragged tensors are implemented through the
        :class:`tf.RaggedTensor` class while the shadow tensors are implemented
        as regular tensors through the :class:`tf.Tensor` class. The shadow
        tensors are regular through padding to the max size. The layers of
        the model must know how to ignore the padding in the computations.

    See :class:`.DLAbstractSequencer`.

    :ivar X: The input data.
    :ivar y: The input reference values.
    :ivar arch: The neural network architecture.
    :vartype arch: :class:`.Architecture`
    :ivar batch_size: The number of elements per batch.
    :vartype batch_size: int
    :ivar total_elems: Maximum number of input elements (i.e., in the full batch)
    :vartype total_elems: int
    :ivar max_depth: The max depth of the model.
    :vartype max_depth: int
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, X, y, batch_size, **kwargs):
        """
        Initialize the member attributes of the DLSparseShadowSequencer.

        :param X: The input data.
        :param y: The input reference values.
        :param batch_size: The number of elements per batch.
        :type batch_size: int
        :param kwargs: The key-word specification to parametrize the
            sequencer.
        """
        # Call parent's init
        super().__init__(X, y, batch_size, **kwargs)
        # Assign values to member attributes
        self.total_elems = len(self.X[0])
        self.max_depth = len(self.X[1][0])
        # Convert X, y to shadow tensors
        self.convert_to_shadow()

    # ---   SEQUENCING MODE METHODS   --- #
    # ----------------------------------- #
    def getitem_training(self, idx):
        """
        See :meth:`.DLAbstractSequencer.getitem_training`.
        """
        # Apply random shuffle, if necessary
        if (
            self.random_shuffle_indices and
            self.Irandom is not None and
            not self.shuffled
        ):
            if isinstance(self.X, list):
                self.apply_random_indices()
            else:
                raise DeepLearningException(
                    'DLSparseShadowSequencer does not support non-list X on'
                    '__getitem__ calls.'
                )
            self.shuffled = True
        # Obtain start and end points for the indexing interval
        max_idx = len(self.X[0]) if isinstance(self.X, list) else len(self.X)
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, max_idx)
        # Extract batch
        batch_X = self.extract_input_batch(start_idx, end_idx)
        batch_y = self.extract_reference_batch(start_idx, end_idx)
        # Return batch
        return batch_X, batch_y

    def getitem_predict(self, idx):
        """
        See :meth:`.DLAbstractSequencer.getitem_predict`.
        """
        # Obtain start and end points for the indexing interval
        max_idx = len(self.X[0]) if isinstance(self.X, list) else len(self.X)
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, max_idx)
        # Extract batch
        batch_X = self.extract_input_batch(start_idx, end_idx)
        # Return batch
        return batch_X

    def on_epoch_end_training(self):
        """
        See :meth:`.DLAbstractSequencer.on_epoch_end_training`.
        """
        # Random index shuffling
        if self.random_shuffle_indices:
            if isinstance(self.X, list):
                if self.Irandom is None:  # First random shuffle of indices
                    self.init_random_indices()
                else:  # After the first random shuffle of indices
                    # Undo previous shuffle
                    self.apply_random_indices()
            else:
                raise DeepLearningException(
                    'DLSparseShadowSequencer does not support non-list X on'
                    'on_epoch_end calls.'
                )
            np.random.shuffle(self.Irandom)  # Shuffle indices
            self.shuffled = False  # Flag to shuffle on first __getitem__ call


    # ---  RANDOM INDEXING METHODS  --- #
    # --------------------------------- #
    def init_random_indices(self):
        """
        See :meth:`.DLAbstractSequencer.init_random_indices`.
        """
        # Number of input point clouds
        m = self.X[0].shape[0] if isinstance(self.X, list) else self.X.shape[0]
        # Initialize random indices
        self.Irandom = np.arange(  # Index for each input pcloud
            m, dtype=np.int32
        )

    def apply_random_indices(self):
        """
        See :meth:`.DLAbstractSequencer.apply_random_indices`.
        """
        tensors_per_pcloud = len(self.X)  # Tensors per input pcloud
        for i in range(tensors_per_pcloud):
            self.X[i] = tf.gather(self.X[i], self.Irandom, axis=0)
        self.y = tf.gather(self.y, self.Irandom, axis=0)

    # ---   SHADOW METHODS   --- #
    # -------------------------- #
    def convert_to_shadow(self):
        """
        Convert the input data (self.X and self.y) to shadow tensors.

        Shadow tensors represent the information of ragged tensors (i.e., those
        with irregular dimensionality) using regular tensors with padding.

        :return: Nothing at all, but the internal state of the
            :class:`.DLSparseShadowSequencer` is updated.
        """
        # Compute element with most rows in the batch
        max_rows = [  # X[1][k][t][0] is hk at depth t for element k
            np.max([
                self.X[1][k][t][0].shape[0] for k in range(self.total_elems)
            ])
            for t in range(self.max_depth)
        ]
        # Compute start row for each element of the batch at each depth
        # to account for padding to avoid RaggedTensor in GPU (due to tf.map_fn)
        start = [[] for k in range(self.total_elems)]
        for k in range(self.total_elems):
            startk = start[k]
            for t in range(self.max_depth):
                max_rows_t = max_rows[t]
                rows = self.X[1][k][t][0].shape[0]
                padding = max_rows_t - rows
                startk.append(padding)
        # Prepare local buffer to store padded data
        X = [[] for _ in range(4)]
        X[0] = [None for k in range(self.total_elems)]
        X[1] = [
            [[None, None] for t in range(self.max_depth)]
            for k in range(self.total_elems)
        ]
        X[2] = [
            [None for t in range(self.max_depth-1)]
            for k in range(self.total_elems)
        ]
        X[3] = [
            [None for t in range(self.max_depth-1)]
            for k in range(self.total_elems)
        ]
        # Pad inputs to have fixed number of rows
        for k in range(self.total_elems):
            startk = start[k]
            for t in range(self.max_depth):
                startkt = startk[t]
                if t == 0:
                    # Pad input features
                    X[0][k] = np.pad(
                        self.X[0][k],
                        [[startkt, 0], [0, 0]],
                        "constant",
                        constant_values=0
                    )
                    if self.y is not None:
                        # Pad reference labels
                        if len(self.y[k].shape) == 1:
                            self.y[k] = np.pad(
                                self.y[k],
                                [startkt+1, 0],
                                "constant",
                                constant_values=-1
                            )
                        else:
                            self.y[k] = np.pad(
                                self.y[k],
                                [[startkt+1, 0], [0, 0]],
                                "constant",
                                constant_values=-1
                            )
                # Pad hk
                pad_vec = [startkt, 0]
                X[1][k][t][0] = np.pad(
                    self.X[1][k][t][0],
                    pad_vec,
                    "constant",
                    constant_values=-1
                )
                # Pad hv
                X[1][k][t][1] = np.pad(
                    self.X[1][k][t][1], pad_vec, "constant", constant_values=0
                )
                # Pad hD
                if t > 0:
                    X[2][k][t-1] = np.pad(
                        self.X[2][k][t-1],
                        pad_vec,
                        "constant",
                        constant_values=-1
                    )
                # Pad hU
                if t < (self.max_depth-1):
                    X[3][k][t] = np.pad(
                        self.X[3][k][t],
                        pad_vec,
                        "constant",
                        constant_values=-1
                    )
        # Update X itself
        self.X = [
          tf.constant(X[0])
        ] + [
          tf.constant([[*X[1][k][t][0]] for k in range(self.total_elems)])
          for t in range(self.max_depth)
        ] + [
          tf.constant([[*X[1][k][t][1]] for k in range(self.total_elems)])
          for t in range(self.max_depth)
        ] + [
          tf.constant([[*X[2][k][t]] for k in range(self.total_elems)])
          for t in range(self.max_depth-1)
        ] + [
          tf.constant([[*X[3][k][t]] for k in range(self.total_elems)])
          for t in range(self.max_depth-1)
        ] + [
          tf.constant([self.X[4][k][:, t] for k in range(self.total_elems)])
          for t in range(self.max_depth)
        ] + [
          tf.constant([start[k][t] for k in range(self.total_elems)])
          for t in range(self.max_depth)
        ]
        if self.y is not None:
            self.y = tf.constant(self.y, dtype=tf.dtypes.float32)
