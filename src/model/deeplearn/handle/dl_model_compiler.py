# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.loss.class_weighted_binary_crossentropy import \
    vl3d_class_weighted_binary_crossentropy
from src.model.deeplearn.loss.class_weighted_categorical_crossentropy import \
    vl3d_class_weighted_categorical_crossentropy
from src.model.deeplearn.loss.ragged_binary_crossentropy import \
    vl3d_ragged_binary_crossentropy
from src.model.deeplearn.loss.ragged_categorical_crossentropy import \
    vl3d_ragged_categorical_crossentropy
from src.model.deeplearn.loss.ragged_class_weighted_binary_crossentropy import \
    vl3d_ragged_class_weighted_binary_crossentropy
from src.model.deeplearn.loss.ragged_class_weighted_categorical_crossentropy \
    import vl3d_ragged_class_weighted_categorical_crossentropy
from src.model.deeplearn.handle.dl_class_weighter import DLClassWeighter
from src.main.main_config import VL3DCFG
import src.main.main_logger as LOGGING
from src.utils.dict_utils import DictUtils
import tensorflow as tf
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class DLModelCompiler:
    """
    Class to compile deep learning models.

    :ivar comp_args: The built compilation arguments (do not confuse with the
        compilation arguments specification typically referred to as
        `compilation_args` instead of `comp_args`).
    :vartype comp_args: dict
    :ivar class_weighter: The class weighter to handle the model's class
        weights, if needed/requested.
    :vartype class_weighter: :class:`.DLClassWeighter`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, compilation_args):
        """
        Initialize a DLModelCompiler.

        :param compilation_args: The specification on how to compile the model.
            See
            :meth:`dl_model_compiler.DLModelCompiler.build_compilation_args`
            and
            :class:`.SimpleDLModelHandler`
            .
        :type compilation_args: dict
        """
        self.comp_args = DLModelCompiler.build_compilation_args(
            compilation_args
        )
        self.class_weighter = DLClassWeighter()

    # ---  COMPILE METHODS  --- #
    # ------------------------- #
    def compile(self, mh, X=None, y=None, y_rf=None, **kwargs):
        """
        The method that provides the logic to compile a model.

        :param mh: The model handler to be compiled.
        :type mh: :class:`.DLModelHandler`
        :param X: Optionally, the coordinates might be used for a better
            initialization (e.g., automatically derive the number of expected
            input points, or the dimensionality of the space where the points
            belong to).
        :param y: Optionally, the labels might be used for a better
            initialization (e.g., automatically derive the number of classes).
        :param y_rf: The expected values for each receptive field. Can be
            used to derive class weights.
        :return: The model handler itself after compiling the architecture,
            which implies modifying its internal state.
        :rtype: :class:`.DLModelHandler`
        """
        # Build architecture
        if not mh.arch.is_built():
            mh.arch.build()
        elif kwargs.get('arch_plot', False):  # If not,
            # at least plot the built architecture if requested
            mh.arch.plot()
        mh.compiled = mh.arch.nn
        # Determine class weights if possible
        class_weight = None
        if y_rf is not None:
            class_weight = self.class_weighter.handle_class_weight(mh, y_rf)
        # Build compilation args
        comp_args = DLModelCompiler.build_compilation_args(
            mh.compilation_args
        )
        if class_weight is not None:  # Recompile for custom class weight loss
            comp_args['loss'] = comp_args['loss'](
                np.array(list(class_weight.values()), dtype=np.float32)
            )
        # Compile
        mh.compiled.compile(
            run_eagerly=VL3DCFG['MODEL']['SimpleDLModelHandler']['run_eagerly'],
            **comp_args
        )
        return mh

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    @staticmethod
    def build_compilation_args(comp_args):
        """
        Build the compilation arguments from given spec.

        :param comp_args: The specification to build the compilation arguments.
        :return: The dictionary of compilation arguments.
        :rtype: dict
        """
        # Build optimizer : Extract args
        opt_args = comp_args['optimizer']
        opt_alg = opt_args['algorithm'].lower()
        opt_lr = opt_args.get('learning_rate', None)
        # Build optimizer : Determine class (algorithm)
        optimizer = None
        if opt_alg == 'sgd':
            optimizer = tf.keras.optimizers.SGD
        if opt_alg == 'adam':
            optimizer = tf.keras.optimizers.Adam
        if optimizer is None:
            raise DeepLearningException(
                'DLModelCompiler cannot compile a model without an '
                'optimizer. None was given.'
            )
        # Build optimizer : Handle learning rate
        if isinstance(opt_lr, dict):  # Learning schedule
            lr_sched_type = opt_lr['schedule']
            if lr_sched_type == 'exponential_decay':
                opt_lr = tf.keras.optimizers.schedules.ExponentialDecay(
                    **opt_lr['schedule_args']
                )
            else:
                raise DeepLearningException(
                    'DLModelCompiler received an unexpected learning '
                    f'rate schedule: "{lr_sched_type}".'
                )
        # Build optimizer
        optimizer = optimizer(**DictUtils.delete_by_val({
            'learning_rate': opt_lr,
        }, None))
        # Build loss : Extract args
        loss_args = comp_args['loss']
        loss_fun = loss_args['function'].lower()
        # Build loss : Determine class (function)
        instantiate_loss = True
        loss = None
        if loss_fun == 'binary_crossentropy':
            loss = tf.keras.losses.BinaryCrossentropy
        if loss_fun == 'ragged_binary_crossentropy':
            loss = vl3d_ragged_binary_crossentropy
        if loss_fun == 'class_weighted_binary_crossentropy':
            loss = vl3d_class_weighted_binary_crossentropy
            instantiate_loss = False  # Instantiate later with class weights
        if loss_fun == 'ragged_class_weighted_binary_crossentropy':
            loss = vl3d_ragged_class_weighted_binary_crossentropy
            instantiate_loss = False  # Instantiate later with class weights
        if loss_fun == 'categorical_crossentropy':
            loss = tf.keras.losses.CategoricalCrossentropy
        if loss_fun == 'ragged_categorical_crossentropy':
            loss = vl3d_ragged_categorical_crossentropy
        if loss_fun == 'class_weighted_categorical_crossentropy':
            loss = vl3d_class_weighted_categorical_crossentropy
            instantiate_loss = False  # Instantiate later with class weights
        if loss_fun == 'ragged_class_weighted_categorical_crossentropy':
            loss = vl3d_ragged_class_weighted_categorical_crossentropy
            instantiate_loss = False  # Instantiate later with class weights
        if loss_fun == 'sparse_categorical_crossentropy':
            loss = tf.keras.losses.SparseCategoricalCrossentropy
        if loss is None:
            raise DeepLearningException(
                'DLModelCompiler cannot compile a model without a loss '
                'function. None was given.'
            )
        # Build loss
        LOGGING.LOGGER.debug(
            f'Compiling deep learning model with "{loss_args["function"]}" '
            f'({loss.__name__}) loss function.'
        )
        if instantiate_loss:
            loss = loss()
        # Build metrics : Extract args
        metrics_args = comp_args['metrics']
        # Build metrics : Determine metrics (list of classes)
        metrics = []
        for metric_name in metrics_args:
            metric_class = None
            if metric_name == 'sparse_categorical_accuracy':
                metric_class = tf.keras.metrics.sparse_categorical_accuracy
            if metric_name == 'categorical_accuracy':
                metric_class = tf.keras.metrics.categorical_accuracy
            if metric_name == 'binary_accuracy':
                metric_class = tf.keras.metrics.binary_accuracy
            if metric_name == 'precision':
                metric_class = tf.keras.metrics.Precision(name='precision')
            if metric_name == 'recall':
                metric_class = tf.keras.metrics.Recall(name='recall')
            if metric_class is None:
                raise DeepLearningException(
                    'DLModelCompiler cannot compile a model because a '
                    f'given metric cannot be interpreted ("{metric_name}").'
                )
            metrics.append(metric_class)
        if len(metrics) < 1:
            LOGGING.LOGGER.debug(
                'DLModelCompiler detected a model compilation with no '
                'metrics. While this is supported, recall an arbitrary number '
                'of evaluation metrics can be used to evaluate the training '
                'performance. These metrics can be more easy to interpret or '
                'bring further insights into the model than the loss function '
                'alone.'
            )
        # Return dictionary of built compilation args
        return {
            'optimizer': optimizer,
            'loss': loss,
            'metrics': metrics
        }
