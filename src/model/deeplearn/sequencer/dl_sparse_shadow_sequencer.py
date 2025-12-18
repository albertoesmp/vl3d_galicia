# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.dlrun.hierarchical_fps_pre_processorpp import \
    HierarchicalFPSPreProcessorPP
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
    :ivar start: The start row-index for each element in the batch. It is used
        to separate padding data from real data. The list of start indices
        is computed when initializing the sequencer.
    :vartype start: list of int
    :ivar max_rows: The max number of rows (i.e., the number of rows of the
        element with most rows) at each depth.
    :vartype max_rows: list of int
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
        self.start = None
        # Prepare the input data
        self.prepare_data()

    # ---   SEQUENCING MODE METHODS   --- #
    # ----------------------------------- #
    def getitem_training(self, idx):
        """
        See :meth:`.DLAbstractSequencer.getitem_training`.
        """
        #start = time.perf_counter()  # TODO Remove : Profiling
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
        #start_extract = time.perf_counter()  # TODO Remove : Profiling
        batch_X = self.extract_input_batch(start_idx, end_idx)
        batch_y = self.extract_reference_batch(start_idx, end_idx)
        #end_extract = time.perf_counter()  # TODO Remove : Profiling
        # Prepare indexing maps (h_{1}, ..., h_{max_depth})
        #start_indexing = time.perf_counter()  # TODO Remove : Profiling
        self.prepare_indexing_maps(start_idx, end_idx, batch_X)
        #end_indexing = time.perf_counter()  # TODO Remove : Profiling
        #end = time.perf_counter()  # TODO Remove : Profiling
        # TODO Remove : Profiling section ---
        """print(f'\nDLSparseShadowSequencer: Batch extraction took {end_extract-start_extract:.3f} seconds.')
        print(f'DLSparseShadowSequencer: Indexing preparation took {end_indexing-start_indexing:.3f} seconds.')
        print(f'DLSparseShadowSequencer: Indexing getitem took {end-start:.3f} seconds.')"""
        # --- TODO Remove : Profiling section
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
        # Prepare indexing maps (h_{1}, ..., h_{max_depth})
        self.prepare_indexing_maps(start_idx, end_idx, batch_X)
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
        m = len(self.X[0]) if isinstance(self.X, list) else self.X.shape[0]
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
            self.X[i] = [self.X[i][j] for j in self.Irandom]
        self.y = [self.y[j] for j in self.Irandom]
        self.start = [self.start[j] for j in self.Irandom]

    # ---   SHADOW METHODS   --- #
    # -------------------------- #
    def prepare_data(self):
        """
        Prepare the input data (self.X and self.y) unrolling the elements per
        depth. Also compute the max number of rows among all elements and the
        start row-index for each element in the batch.

        :return: Nothing at all, but the internal state of the
            :class:`.DLSparseShadowSequencer` is updated.
        """
        # Compute element with most rows in the batch
        self.max_rows = [  # X[1][k][t][0] is hk at depth t for element k
            np.max([
                self.X[1][k][t][0].shape[0] for k in range(self.total_elems)
            ])
            for t in range(self.max_depth)
        ]
        # Copy references if given, to avoid modifying original references
        # Note this might not be necessary depending on frameworks' integration
        if self.y is not None:
            self.y = [  # Work on copy to avoid padding original references
                np.array(yi, dtype=yi.dtype) for yi in self.y
            ]
        # Compute start row for each element of the batch at each depth
        # to account for padding to avoid RaggedTensor in GPU (due to tf.map_fn)
        self.start = [[] for k in range(self.total_elems)]
        for k in range(self.total_elems):
            startk = self.start[k]
            for t in range(self.max_depth):
                max_rows_t = self.max_rows[t]
                rows = self.X[1][k][t][0].shape[0]
                padding = max_rows_t - rows
                startk.append(padding)
        self.start = np.array(self.start, dtype=np.int32)
        # Update X itself
        self.X = [
            self.X[0]
        ] + [
                [
                    self.X[1][k][t][0].astype(np.int32)
                    for k in range(self.total_elems)
                ]
                for t in range(self.max_depth)
        ] + [
                [
                    self.X[1][k][t][1].astype(np.int32)
                    for k in range(self.total_elems)
                ]
                for t in range(self.max_depth)
        ] + [
                [
                    self.X[2][k][t].astype(np.int32)
                    for k in range(self.total_elems)
                ]
                for t in range(self.max_depth-1)
        ] + [
                [
                    self.X[3][k][t].astype(np.int32)
                    for k in range(self.total_elems)
                ]
                for t in range(self.max_depth-1)
        ] + [
                [
                    self.X[4][k][:, t].astype(np.int32)
                    for k in range(self.total_elems)
                ]
                for t in range(self.max_depth)
        ] + [
                [
                    self.start[k][t]
                    for k in range(self.total_elems)
                ]
                for t in range(self.max_depth)
        ]

    def prepare_indexing_maps(self, start_idx, end_idx, batch_X):
        """
        Update the given batch (``batch_X``) so the indexing maps
        :math:`(h_1, \ldots, h_{t^*})`, :math:`(h^D_1, \ldots, h^D_{t^*-1}`,
        and :math:`(h^U_1, \ldots, h^U_{t^*-1}` apply the corresponding offset
        such that the batch can be unified into a single hash table. This
        property can be achieved by making sure that the index zero of the
        second element is transformed to be the last plus one index of the
        first element and applying this inductively to all subsequent elements
        in the batch.

        :param start_idx: The start index for the current batch (inclusive).
        :param end_idx: The end index for the current batch (exclusive)
        :param batch_X: The current batch (note references aer not needed here).
        :return: Nothing at all but the indexing maps are updated in place, i.e,
            inside the current batch (``batch_X``).
        """
        # Prepare indexing maps (h_{1}, ..., h_{max_depth})
        hk_start, hD_start, hU_start = 0, 2*self.max_depth, 3*self.max_depth-1
        cbs = end_idx-start_idx # Current batch size (<= requested batch size)
        premax_depth = self.max_depth - 1
        for t in range(self.max_depth):
            # Advance indices to handle (k, v) for h map at depth t
            hk_start += 1
            hD_start += 1 if t < premax_depth else 0
            hU_start += 1 if t > 0 else 0
            hkt = batch_X[hk_start]
            k_offset = 0  # Offset for first batch is zero
            # For each vector in the batch of keys or values
            for k in range(cbs):
                hktk = hkt[k]  # Obtain k-th batch
                mask = hktk > -1  # Find non-shadow indices
                hktk[mask] = hktk[mask] + k_offset  # Update non-shadow indices
                # Update downsampling indexing
                if t < premax_depth:
                    hDtk = batch_X[hD_start][k]
                    mask = hDtk > -1  # Find non-shadow indices
                    hDtk[mask] = hDtk[mask] + k_offset  # Update non-shadow
                # Update upsampling indexing
                if t > 0:
                    hUtk = batch_X[hU_start][k]
                    mask = hUtk > -1  # Find non-shadow indices
                    hUtk[mask] = hUtk[mask] + k_offset  # Update non-shadow
                k_offset = max(k_offset, np.max(hktk)+1)  # Increment offset

    # ---  BATCH EXTRACTION METHODS  --- #
    # ---------------------------------- #
    def extract_input_batch(self, start_idx, end_idx):
        """
        See :meth:`.DLAbstractSequencer.extract_input_batch`.
        """
        X_batch = [Xi[start_idx:end_idx] for Xi in self.X]
        batch_elems = end_idx - start_idx
        for k in range(batch_elems):
            startk = self.start[start_idx+k]
            # Pad input features
            X_batch[0][k] = np.pad(
                X_batch[0][k],
                [[startk[0], 0], [0, 0]],
                "constant",
                constant_values=0
            )
            for t in range(self.max_depth):
                startkt = startk[t]
                # Pad hk
                pad_vec = [startkt, 0]
                hk_idx = 1+t
                X_batch[hk_idx][k] = np.pad(
                    X_batch[hk_idx][k].astype(np.int32),
                    pad_vec,
                    "constant",
                    constant_values=-1
                )
                # Pad hv
                hv_idx = hk_idx + self.max_depth
                X_batch[hv_idx][k] = np.pad(
                    X_batch[hv_idx][k].astype(np.int32),
                    pad_vec,
                    "constant",
                    constant_values=0
                )
                # Pad hD and hU
                if t < (self.max_depth-1):
                    # Pad hD
                    hD_idx = hv_idx+self.max_depth
                    X_batch[hD_idx][k] = np.pad(
                        X_batch[hD_idx][k].astype(np.int32),
                        [startk[t+1], 0],
                        "constant",
                        constant_values=-1
                    )
                    # Pad hU
                    hU_idx = hD_idx + self.max_depth - 1
                    X_batch[hU_idx][k] = np.pad(
                        X_batch[hU_idx][k].astype(np.int32),
                        pad_vec,
                        "constant",
                        constant_values=-1
                    )
        return [np.array(Xi) for Xi in X_batch]

    def extract_reference_batch(self, start_idx, end_idx):
        """
        See :meth:`.DLAbstractSequencer.extract_reference_batch`.
        """
        y_batch = self.y[start_idx:end_idx]
        batch_elems = end_idx - start_idx
        for k in range(batch_elems):
            startk = self.start[start_idx+k][0]
            # Pad reference labels
            if len(y_batch[k].shape) == 1:
                y_batch[k] = np.pad(
                    y_batch[k].astype(np.int32),
                    [startk+1, 0],
                    "constant",
                    constant_values=-1
                )
            else:
                y_batch[k] = np.pad(
                    y_batch[k].astype(np.int32),
                    [[startk+1, 0], [0, 0]],
                    "constant",
                    constant_values=-1
                )

        return np.array(y_batch)
