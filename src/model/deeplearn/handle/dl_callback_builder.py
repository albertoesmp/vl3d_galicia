# ---   IMPORT   --- #
# ------------------ #
from src.main.main_config import VL3DCFG
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
        # Handle check point call back
        if mh.path_manager.checkpoint_path is not None:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                mh.path_manager.checkpoint_path,
                monitor=mh.checkpoint_monitor,
                save_best_only=True,
                save_weights_only=True
            ))
        # Handle learning rate on plateau callback
        if mh.learning_rate_on_plateau is not None:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                **mh.learning_rate_on_plateau
            ))
        # Handle early stopping callback
        if mh.early_stopping is not None:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                **mh.early_stopping
            ))
        # Handle tensorboard callback
        MHCFG = VL3DCFG['MODEL']['SimpleDLModelHandler']
        logdir = MHCFG['tensorboard_log_dir']
        if logdir is not None:
            callbacks.append(tf.keras.callbacks.TensorBoard(
                log_dir=logdir,
                histogram_freq=MHCFG['tensorboard_histogram_frequency'],
                profile_batch=tuple(MHCFG['tensorboard_profile_batch'])
            ))
        return callbacks
