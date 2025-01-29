# ---   IMPORTS   --- #
# ------------------- #
import tensorflow as tf
import numpy as np
from abc import abstractmethod


# ---   CLASS   --- #
# ----------------- #
class DLAbstractSequencer(tf.keras.utils.Sequence):
    """
    :author: Alberto M. Esmoris Pena

    A deep learning sequencer governs how the input data is fed to a neural
    network. The abstract sequencer for deep learning models provides common
    logic for different sequencers. It cannot be used directly because it is
    an abstract class, hence it must be extended by a concrete implementation.

    :ivar X: The input data.
    :ivar y: The input reference values.
    :ivar arch: The neural network architecture.
    :vartype arch: :class:`.Architecture`
    :ivar batch_size: The number of elements per batch.
    :vartype batch_size: int
    :ivar random_shuffle_indices: Flag governing whether the indexing of the
        elements (i.e., the order in which they are given) must be shuffled
        or not (typically at the end of each epoch).
    :vartype random_shuffle_indices: bool
    :ivar getitem_method: The method that must be called by
        :meth:`.DLAbstractSequencer.__getitem__`. It might change depending
        on whether training or predictive mode is enabled.
    :ivar on_epoch_end_method: The method that must be called by
        :meth:`.DLAbstractSequencer.on_epoc_end`. It might change depending
        on whether training or perdictive mode is enabled.
    :ivar Irandom: A cache for the random shuffle of elements' indices.
    :ivar shuffled: A cache to track whether the elements have been shuffled
        (True) or not (False).
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, X, y, batch_size, **kwargs):
        """
        Initialize the member attributes of the DLAbstractSequencer.

        :param X: The input data.
        :param y: The input reference values.
        :param batch_size: The number of elements per batch.
        :type batch_size: int
        :param kwargs: The key-word specification to parametrize the
            sequencer.
        """
        # Call parent's init
        super().__init__()
        # Assign values to member attributes
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.arch = kwargs.get('arch', None)
        self.random_shuffle_indices = kwargs.get(
            'random_shuffle_indices', False
        )
        if kwargs.get('training', True):
            self.enable_training_mode()
        else:
            self.enable_predict_mode()
        # Cache attributes
        self.Irandom = None
        self.shuffled = False

    # ---   SEQUENCE METHODS   --- #
    # ---------------------------- #
    def __len__(self):
        """
        Obtain the number of input batches.

        :return: Number of input batches.
        :rtype: int
        """
        if isinstance(self.X, list):
            return int(np.ceil(len(self.X[0]) / self.batch_size))
        else:
            return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        """
        Obtain the batch corresponding to the given index.

        Note that this method can work for training, in which case it calls
        the :meth:`.DLAbstractSequencer.getitem_training` method (or its
        child/derived class override), but it can also work for predictions,
        in which case it calls the :meth:`.DLAbstractSequencer.getitem_predict`.

        See :meth:`.DLAbstractSequencer.getitem_training`,
        :meth:`.DLAbstractSequencer.enable_training_mode`,
        See :meth:`.DLAbstractSequencer.getitem_predict`, and
        :meth:`.DLAbstractSequencer.enable_predict_mode`,

        :param idx: The index identifying the batch that must be obtained.
        :type idx: int
        :return: The batch corresponding to the given index as a tuple
            (X, y) for training and simply X for predictions.
        """
        return self.getitem_method(idx)

    def on_epoch_end(self):
        """
        Logic to handle the dataset between epochs.

        Note that this method can work for training, in which case it calls
        the :meth:`.DLAbstractSequencer.on_epoch_training` method (or its
        child/derived class override), but it can also work for predictions,
        in which case it calls the
        :meth:`.DLAbstractSequencer.on_epoch_predict`.

        See :meth:`.DLAbstractSequencer.on_epoch_end_training`,
        :meth:`.DLAbstractSequencer.enable_training_mode`,
        See :meth:`.DLAbstractSequencer.on_epoch_end_predict`, and
        :meth:`.DLAbstractSequencer.enable_predict_mode`,

        :return: Nothing at all, but the internal state of the sequencer is
            updated.
        """
        return self.on_epoch_end_method()

    # ---   SEQUENCING MODE METHODS   --- #
    # ----------------------------------- #
    @abstractmethod
    def getitem_training(self, idx):
        """
        Method that provides the logic for
        :meth:`.DLAbstractSequencer.__getitem__` in training contexts.
        It must be overridden by any derived class of
        :class:`.DLAbstractSequencer` that aims to being concrete
        (non-abstract, i.e., actually callable).

        :param idx: The index identifying the batch that must be obtained.
        :type idx: int
        :return: The batch corresponding to the given index as a tuple (X, y).
        :rtype: tuple
        """
        pass

    def on_epoch_end_training(self):
        """
        Method that provides the logic for
        :meth:`.DLAbstractSequencer.on_epoch_end` in training contexts.
        It can be overridden by any derived class of
        :class:`.DLAbstractSequencer` to change ts behavior.

        The default logic to handle the dataset between epochs provided by
        :class:`.DLAbstractSequencer` is the random shuffle of the data so
        the input batches are given in a different order at each training
        epochs.

        See :meth:`.DLAbstractSequencer.init_random_indices` and
        :meth:`.DLAbstractSequencer.apply_random_indices` methods.

        :return: Nothing at all, but the internal state of the sequencer is
            updated.
        """
        # Random index shuffling
        if self.random_shuffle_indices:
            if self.Irandom is None:  # First random shuffle of indices
                self.init_random_indices()
            else:  # After the first random shuffle of indices
                self.apply_random_indices()
            np.random.shuffle(self.Irandom)  # Shuffle indices
            self.shuffled = False  # Flag to shuffle on first __getitem__ call

    def enable_training_mode(self):
        """
        Enable the training mode of the sequencer so it can be used for
        training.

        See :meth:`.DLAbstractSequencer.__getitem__` and
        :meth:`.DLAbstractSequencer.getitem_training`.

        :return: Nothing at all but the :class:`.DLAbstractSequencer` state is
            updated to operate on training mode.
        """
        self.getitem_method = self.getitem_training
        self.on_epoch_end_method = self.on_epoch_end_training

    @abstractmethod
    def getitem_predict(self, idx):
        """
        Method that provides the logic for
        :meth:`.DLAbstractSequencer.__getitem__` in prediction contexts.
        It must be overridden by any derived class of
        :class:`.DLAbstractSequencer` that aims to being concrete
        (non-abstract, i.e., actually callable).

        :param idx: The index identifying the batch that must be obtained.
        :type idx: int
        :return: The batch corresponding to the given index but without
            references.
        """
        pass

    def on_epoch_end_predict(self):
        """
        Method that provides the logic for
        :meth:`.DLAbstractSequencer.on_epoch_end` in prediction contexts.
        It cab be overridden by any derived class of
        :class:`.DLAbstractSequencer` to change its behavior.

        The default logic to handle the dataset between epochs provided by
        :class:`.DLAbstractSequencer` is to do nothing.

        :return: Nothing at all, but the internal state of the sequencer is
            updated.
        """
        pass

    def enable_predict_mode(self):
        """
        Enable the predict mode of the sequencer so it can be used for
        predictions.

        See :meth:`.DLAbstractSequencer.__getitem__` and
        :meth:`.DLAbstractSequencer.getitem_predict`.

        :return: Nothing at all but the :class:`.DLAbstractSequencer` state is
            updated to operate on predict mode.
        """
        self.getitem_method = self.getitem_predict
        self.on_epoch_end_method = self.on_epoch_end_predict

    # ---  RANDOM INDEXING METHODS  --- #
    # --------------------------------- #
    @abstractmethod
    def init_random_indices(self):
        """
        Method that must provide the logic to initialize the random indices to
        shuffle the data.

        See :meth:`.DLAbstractSequencer.on_epoch_end_training`.

        :return: Nothing at all but the internal state of the
            :class:`.DLAbstractSequencer` is updated (e.g., the self.Irandom
            cache).
        """
        pass

    @abstractmethod
    def apply_random_indices(self):
        """
        Method that must provide the logic to apply the random shuffle based
        on the random indices computed by the
        :meth:`.DLAbstractSequencer.init_random_indices` method.

        See :meth:`.DLAbstractSequencer.on_epoch_end_training` and
        :meth:`.DLAbstractSequencer.init_random_indices`.

        :return: Nothing at all but the internal state of the
            :class:`.DLAbstractSequencer` is updated (e.g., the self.X and
            self.y data).
        """
        pass

    # ---  BATCH EXTRACTION METHODS  --- #
    # ---------------------------------- #
    def extract_input_batch(self, start_idx, end_idx):
        """
        Extract the input batch inside the given indexing interval.

        Note that this method can be overridden by any derived class of
        :class:`.DLAbstractSequencer` that needs to change its behavior.

        :param start_idx: Start point of the indexing interval (inclusive).
        :param end_idx: End point of the indexing interval (exclusive).
        :return: The extracted input batch inside the indexing interval.
        :rtype: list or :class:`np.ndarray`
        """
        if isinstance(self.X, list):
            return [np.array(Xi[start_idx:end_idx]) for Xi in self.X]
        else:
            return np.array(self.X[start_idx:end_idx])

    def extract_reference_batch(self, start_idx, end_idx):
        """
        Extract the reference batch inside the given indexing interval.

        Note that this method can be overridden by any derived class of
        :class:`.DLAbstractSequencer` that needs to change its behavior.

        :param start_idx: Start point of the indexing interval (inclusive).
        :param end_idx: End point of the indexing interval (exclusive).
        :return: The extracted reference batch inside the indexing interval.
        :rtype: :class:`np.ndarray`
        """
        return np.array(self.y[start_idx:end_idx])
