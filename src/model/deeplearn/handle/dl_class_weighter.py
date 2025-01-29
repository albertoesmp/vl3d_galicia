# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
import src.main.main_logger as LOGGING
import numpy as np

# ---   CLASS   --- #
# ----------------- #
class DLClassWeighter:
    """
    Class to handle class weights for deep learning models.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        pass

    # ---  CLASS WEIGHT METHODS  --- #
    # ------------------------------ #
    def handle_class_weight(self, mh, y):
        r"""
        Handle the class weight parameter of a deep learning model's handler.

        If no class weight is requested, then class weight will be None.

        If automatic class weight is requested (i.e., "auto"), then the
        class weight is automatically determined from the distribution of
        expected classes to give a greater weight to less frequent classes
        and a smaller weight to more frequent classes. More concretely, let
        :math:`m` be the number of samples, :math:`m_i` be the number of
        samples corresponding to class :math:`i`, and :math:`n` be the number
        of classes. Thus, each class weight will be :math:`w_i = m/(n m_i)`.

        If class weight is a list, tuple or array of weights it will be
        translated to a dictionary such that the first element is the weight
        for the first class, and so on.

        :param mh: The model handler whose class weights must be handled.
        :param y: The vector of expected point-wise classes.
        :type y: :class:`np.ndarray`
        :return: Class weight prepared for the model.
        """
        # No class weight specification
        if mh.class_weight is None:
            return None
        # Handle class weight specification
        class_weight_low = mh.class_weight.lower() \
            if isinstance(mh.class_weight, str) else mh.class_weight
        if class_weight_low == "auto":  # Automatic
            num_classes = getattr(mh.arch, "num_classes", None)
            if num_classes is None:
                raise DeepLearningException(
                    'DLClassWeighter does not support automatic class '
                    'weight for current architecture: '
                    f'"{mh.arch.__class__.__name__}"'
                )
            keys = [class_id for class_id in range(num_classes)]
            num_samples = np.prod(y.shape)
            num_samples_per_class = np.array([
                np.count_nonzero(y == class_id) for class_id in keys
            ], dtype=int)
            vals = num_samples / num_samples_per_class / num_classes
            class_weight_dict = dict(zip(keys, vals))
            LOGGING.LOGGER.debug(
                'DLClassWeighter automatically generated the '
                f'following dictionary of class weights:\n{class_weight_dict}'
            )
            return class_weight_dict
        else:  # User-given
            return dict(zip(  # List to dict with serial int key
                np.arange(len(mh.class_weight), dtype=int),
                mh.class_weight
            ))
