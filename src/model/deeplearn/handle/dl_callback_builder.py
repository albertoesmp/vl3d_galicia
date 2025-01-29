# ---   IMPORT   --- #
# ------------------ #
import tensorflow as tf

# ---   CLASS   --- #
# ----------------- #
class DLCallbackBuilder:
    """
    Class to handle the building of callbacks for deep learning models.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        pass

    # ---   BUILDING METHODS   --- #
    # ---------------------------- #
    def build(self, mh):
        """
        Build the callbacks for the model handled by the given model handler
        (mh).

        :param mh: The model handler whose callbacks must be built.
        :type mh: :class:`.DLModelHandler` and :class:`.SimpleDLModelHandler`
        :return: The built callbacks
        :rtype: list
        """
        callbacks = []
        if mh.path_manager.checkpoint_path is not None:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                mh.path_manager.checkpoint_path,
                monitor=mh.checkpoint_monitor,
                save_best_only=True,
                save_weights_only=True
            ))
        if mh.learning_rate_on_plateau is not None:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                **mh.learning_rate_on_plateau
            ))
        if mh.early_stopping is not None:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                **mh.early_stopping
            ))
        return callbacks
