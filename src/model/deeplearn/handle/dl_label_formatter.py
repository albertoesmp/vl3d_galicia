# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.arch.spconv3d_pwise_classif import SpConv3DPwiseClassif
from sklearn.preprocessing import LabelBinarizer
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class DLLabelFormatter:
    """
    Class to handle the labels for deep learning models.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        pass

    # ---  LABEL FORMAT METHODS  --- #
    # ------------------------------ #
    def handle_labels_format(self, mh, y):
        """
        Handles the format in which labels must be given to the model.

        For instance, if categorical cross entropy is used, labels must be
        given using one-hot-encoding. However, if sparse categorical cross
        entropy is used, labels must be given as an integer.

        :param mh: The model handler whose labels must be formatted.
        :type mh: :class:`.DLModelHandler`
        :param y: The labels to be formatted.
        :return: The labels prepared for the model.
        """
        # Extract loss function name
        loss_low = mh.compilation_args['loss']['function'].lower()
        # Handle loss functions that demand one-hot labels
        if (
                loss_low == 'categorical_crossentropy' or
                loss_low == 'class_weighted_categorical_crossentropy' or
                loss_low == 'ragged_categorical_crossentropy' or
                loss_low == 'ragged_class_weighted_categorical_crossentropy'
        ):  # Handle one hot encoding for labels
            num_classes = getattr(mh.arch, "num_classes", None)
            if num_classes is None:
                raise DeepLearningException(
                    'DLLabelFormatter does not support categorical or '
                    'binary crossentropy without a priori specifying the '
                    'number of classes.'
                )
            label_binarizer = LabelBinarizer().fit([
                class_id for class_id in range(num_classes)
            ])
            new_y = []
            for i in range(len(y)):
                new_y.append(label_binarizer.transform(y[i].flatten()).astype(
                    y[i].dtype
                ))
            if isinstance(mh.arch, SpConv3DPwiseClassif):
                y = new_y
            else:
                y = np.array(new_y, dtype=y.dtype)
        if (
                loss_low == 'sparse_categorical_crossentropy' and
                mh.class_weight is not None
        ):
            raise DeepLearningException(
                'DLLabelFormatter detected that class weight is requested '
                'for a sparse categorical crossentropy loss. Currently, this '
                'is not supported.'
            )
        if (
                loss_low == 'binary_crossentropy' and
                mh.class_weight is not None
        ):
            raise DeepLearningException(
                'DLLabelFormatter detected that class weight is requested '
                'for a binary crossentropy loss. This is not supported. '
                'Please, use "class_weighted_binary_crossentropy" loss '
                'function instead.'
            )
        # By default, labels can be used straight forward
        return y
