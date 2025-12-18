# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.arch.spconv3d_pwise_classif import SpConv3DPwiseClassif
from src.model.deeplearn.handle.dl_label_formatter import DLLabelFormatter
from src.model.deeplearn.handle.dl_model_handler import DLModelHandler
from src.model.deeplearn.sequencer.dl_sequencer import DLSequencer
from src.model.deeplearn.sequencer.dl_sparse_shadow_sequencer import \
    DLSparseShadowSequencer
from src.utils.preds.prediction_reducer import PredictionReducer
from src.utils.preds.prediction_reducer_factory import PredictionReducerFactory
from src.utils.dict_utils import DictUtils
from src.inout.io_utils import IOUtils
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.handle.dl_model_compiler import DLModelCompiler
from src.model.deeplearn.handle.dl_path_manager import DLPathManager
from src.model.deeplearn.handle.dl_callback_builder import DLCallbackBuilder
from src.model.deeplearn.handle.dl_model_reporter import DLModelReporter
from src.model.deeplearn.sequencer.dl_abstract_sequencer import \
    DLAbstractSequencer
from src.main.main_config import VL3DCFG
import src.main.main_logger as LOGGING
import tensorflow as tf
from tensorflow.python.framework.errors_impl import ResourceExhaustedError as \
    TFResourceExhaustedError, InternalError as TFInternalError
import numpy as np
import copy
import time


# ---   CLASS   --- #
# ----------------- #
class SimpleDLModelHandler(DLModelHandler):
    """
    Class to handle deep learning models in a simple way. It can be seen as the
    baseline deep learning model handler. See :class:`.DLModelHandler`.

    :ivar out_prefix: The output prefix for path expansions, when necessary.
    :vartype out_prefix: str
    :ivar training_epochs: The number of training epochs for fitting the model.
    :vartype training_epochs: int
    :ivar batch_size: The batch size governing the model's input.
    :vartype batch_size: int
    :ivar history: By default None. It will be updated to contain the training
        history when calling fit.
    :vartype history: None or :class:`tf.keras.callbacks.History`
    :ivar checkpoint_monitor: The name of the metric to choose the best
        model. By default, it is "loss", which represents the loss function.
    :vartype checkpoint_monitor: str
    :ivar learning_rate_on_plateau: The key-word arguments governing the
        instantiation of the learning rate on plateau callback.
    :vartype learning_rate_on_plateau: dict
    :ivar early_stopping: The key-word arguments governing the instantiation
        of the early stopping callback.
    :vartype early_stopping: dict
    :ivar sequencer_spec: The specification on how to build the sequencer
        for the input data during model training. See
        :meth:`SimpleDLModelHandler.build_sequencer`.
    :vartype sequencer_spec: dict
    :ivar fit_verbose: Whether to use silent mode (0), show a progress bar (1),
        or print one line per epoch (2). Alternatively, "auto" can be used
        which typically means (1).
    :vartype fit_verbose: str or int
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, arch, **kwargs):
        """
        Initialize/instantiate a simple deep learning model handler.

        See :class:`.DLModelHandler` and
        :meth:`dl_model_handler.DLModelHandler.__init__`.
        """
        # Call parent's init
        super().__init__(arch, **kwargs)
        # Set defaults from VL3DCFG
        kwargs = DictUtils.add_defaults(
            kwargs,
            VL3DCFG['MODEL']['SimpleDLModelHandler']
        )
        # Assign member attributes
        self.path_manager = DLPathManager(model_handling=kwargs)
        self.out_prefix = kwargs.get('out_prefix', None)
        self.training_epochs = kwargs.get('training_epochs', 100)
        self.batch_size = kwargs.get('batch_size', 16)
        self.history = kwargs.get('history', None)
        self.checkpoint_monitor = kwargs.get('checkpoint_monitor', 'loss')
        self.learning_rate_on_plateau = kwargs.get(
            'learning_rate_on_plateau',
            None
        )
        self.early_stopping = kwargs.get('early_stopping', None)
        self.sequencer_spec = kwargs.get(
            'sequencer_spec', kwargs.get('training_sequencer', None)
        )
        self.fit_verbose = kwargs.get('fit_verbose', "auto")
        self.predict_verbose = kwargs.get('predict_verbose', "auto")
        self.prediction_reducer = kwargs.get(
            'prediction_reducer',
            PredictionReducer()
        )
        if isinstance(self.prediction_reducer, dict):
            self.prediction_reducer = \
                PredictionReducerFactory.make_from_dict(
                    self.prediction_reducer
                )
        self.skip_fit_on_zero_epochs = VL3DCFG['MODEL'][
            'SimpleDLModelHandler'
        ].get('skip_fit_on_zero_epochs', False)

    # ---   MODEL HANDLER   --- #
    # ------------------------- #
    def _fit(self, X, y):
        """
        See :class:`.DLModelHandler` and
        :meth:`dl_model_handler.DLModelHandler._fit`.
        """
        # Report the model
        dl_model_reporter = DLModelReporter()
        dl_model_reporter.handle_model_summary_report(self)
        # Check whether skip fit on zero epochs is requested
        if self.skip_fit_on_zero_epochs and self.training_epochs < 1:
            return self
        # Handle training callbacks
        callbacks = self.build_callbacks()
        # Fit the model
        start = time.perf_counter()
        X_rf, y_rf = self.arch.run_pre({
            'X': X,
            'y': y,
            'training_support_points': True
        })
        self.compile(y_rf=y_rf)  # Recompile with y_rf for class weights
        y_rf = DLLabelFormatter().handle_labels_format(  # Depends on loss
            self, y_rf
        )
        self.fit_logic(X_rf, y_rf, callbacks)
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'Deep learning model trained on {len(y_rf)} cases during '
            f'{self.training_epochs} epochs in {end-start:.3f} seconds.'
        )
        # Take best model from checkpoint
        if self.path_manager.checkpoint_path is not None:
            try:
                IOUtils.validate_path_to_file(
                    self.path_manager.checkpoint_path,
                    msg='Cannot find DL model checkpoint at:'
                )
                self.compiled.load_weights(self.path_manager.checkpoint_path)
                LOGGING.LOGGER.debug(
                    'SimpleDLModelHandler restored weights from '
                    f'"{self.path_manager.checkpoint_path}".'
                )
            except FileNotFoundError as fnferr:
                LOGGING.LOGGER.warning(
                    'SimpleDLModelHandler failed to restore DL model weights '
                    f'from "{self.path_manager.checkpoint_path}".'
                )
        # Report and plot history
        dl_model_reporter.handle_history_plots_and_reports(self)
        # Predictions on the training receptive fields for plots and reports
        dl_model_reporter.handle_receptive_fields_plots_and_reports(
            self, X_rf, X=X[0] if isinstance(X, list) else X, y=y
        )
        # Return
        return self

    def fit_logic(self, X, y_rf, callbacks):
        """
        Implement the fit logic.

        :param X: The pre-processed input.
        :param y_rf: The point-wise labels for each receptive field.
        :param callbacks: The callbacks to be executed during fit.
        :return: The fit history (also assigned internally to self.history).
        """
        # Map for caching stuff during fit logic
        fit_cache_map = {
            'fsl_dir_path': self.path_manager.feat_struct_repr_dir,
            'rbf_dir_path': self.path_manager.rbf_feat_extract_repr_dir,
            'rbf_feat_processing_dir_path':
                self.path_manager.rbf_feat_processing_repr_dir,
            'kpconv_representation_dir':
                self.path_manager.kpconv_representation_dir,
            'skpconv_representation_dir':
                self.path_manager.skpconv_representation_dir,
            'lkpconv_representation_dir':
                self.path_manager.lkpconv_representation_dir,
            'slkpconv_representation_dir':
                self.path_manager.slkpconv_representation_dir,
            'out_prefix': self.out_prefix,
            'X': X,
            'y_rf': y_rf,
            'training_epochs': self.training_epochs,
            'callbacks': callbacks,
            'batch_size': self.batch_size,
            'compilef': lambda _y_rf: self.compile(y_rf=_y_rf)
        }
        # Pre-fit logic
        if hasattr(self.arch, 'prefit_logic_callback'):
            self.arch.prefit_logic_callback(fit_cache_map)
        # Fit logic
        input_X, input_y = self.build_sequencer(X, y_rf, True)
        self.history = self.compiled.fit(
            input_X, input_y,
            epochs=self.training_epochs,
            callbacks=callbacks,
            batch_size=self.batch_size,
            verbose=self.fit_verbose
        )
        # Post-fit logic
        if hasattr(self.arch, 'posfit_logic_callback'):
            # Update callbacks for freeze training iterations
            callbacks = self.build_callbacks()
            fit_cache_map['callbacks'] = callbacks
            fit_cache_map['history'] = self.history
            self.arch.posfit_logic_callback(fit_cache_map)
            self.history = fit_cache_map['history']
        return self.history

    def _predict(self, X, y=None, zout=None, plots_and_reports=True):
        """
        See :class:`.DLModelHandler` and
        :meth:`dl_model_handler.DLModelHandler._predict`.
        """
        # Obtain receptive fields
        X_rf = self.arch.run_pre({
            'X': X,
            'support_points': True,
            'plots_and_reports': plots_and_reports
        })
        # Compute predictions on receptive fields
        try:
            predict_rf_start = time.perf_counter()
            zhat_rf = self.predict_rf(X_rf)
            predict_rf_end = time.perf_counter()
        except (TFResourceExhaustedError, TFInternalError) as tferr:
            m = len(X_rf[0]) if isinstance(X_rf, list) else len(X_rf)
            LOGGING.LOGGER.debug(
                'SimpleDLModelHandler could not compute predictions for '
                f'{m} points using the GPU.\n'
                'Trying CPU instead ...'
            )
            with tf.device("cpu:0"):
                predict_rf_start = time.perf_counter()
                zhat_rf = self.predict_rf(X_rf)
                predict_rf_end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'Predictions computed on {len(zhat_rf)} receptive fields '
            f'in {predict_rf_end-predict_rf_start:.3f} seconds.'
        )
        # Bring predictions from receptive fields back to the original input
        zhat = self.arch.run_post(
            {'X': X, 'z': zhat_rf},
            reducer=self.prediction_reducer
        )
        if zout is not None:  # When z is not None it must be a list
            zout.append(zhat)  # Append propagated zhat to z list
        # Final predictions
        yhat, zhat = self.prediction_reducer.select(zhat), None
        # Do plots and reports
        if plots_and_reports:
            if isinstance(self.arch, SpConv3DPwiseClassif):
                _X_rf = [
                    rfi.compute_active_centroids(0)
                    for rfi in self.arch.pre_runnable.last_call_receptive_fields
                ]
                _F_rf = X_rf[0]
            else:
                _X_rf = X_rf[0] if isinstance(X_rf, list) else X_rf
                _F_rf = X_rf[1] if isinstance(X_rf, list) else None
            DLModelReporter().do_receptive_fields_plots_and_reports(
                mh=self,
                X_rf=_X_rf,
                zhat_rf=zhat_rf,
                X=X[0] if isinstance(X, list) else X,
                y=y,
                F_rf=_F_rf,
                training=False
            )
        # Return
        return yhat

    def predict_rf(self, X_rf):
        """
        Compute the predictions on the given receptive fields.

        :param X_rf: The receptive fields that must be predicted. It can be
            a list with the different inputs (e.g., for hierarchical models
            that use a hierarchical pre-processor like
            :class:`.HierarchicalFPSPreProcessorPP` or
            :class:`.HierarchicalSGPreProcessorPP`) or
            directly an input :class:`np.ndarray` (e.g., input structure space).
        :type X_rf: list or :class:`np.ndarray`
        :return: The predictions as directly computed by the model (e.g.,
            the softmax probabilities from a point-wise classifier).
        :rtype: :class:`np.ndarray`
        """
        # Build sequencer
        input_X, input_y = self.build_sequencer(X_rf, None, False)
        # Predict through sequencer
        if isinstance(input_X, DLAbstractSequencer):
            # TODO Rethink : X_rf mem. can be duplicated (especially for SpConv)
            LOGGING.LOGGER.info(
                f'Predicting through sequencer in {len(input_X)} batches ...'
            )
            zhat_rf = []
            for x in input_X:
                zhat_rf.append(self.compiled.predict_on_batch(x))
            zhat_rf = np.vstack(zhat_rf)
            # TODO Rethink : Handle shadow receptive fields ---
            # TODO Rethink : Move to sequencer logic?
            # Handle shadow receptive fields (i.e., remove padding from output)
            if isinstance(self.arch, SpConv3DPwiseClassif):
                start = input_X.X[-input_X.max_depth]
                zhat_rf = [
                    zhat_rfi[1 + start[i]:]
                    for i, zhat_rfi in enumerate(zhat_rf)
                ]
            # --- TODO Rethink : Handle shadow receptive fields
        # Predict without sequencer (raw data)
        else:
            LOGGING.LOGGER.info(
                f'Predicting without sequencer ...'
            )
            zhat_rf = self.compiled.predict(
                input_X,
                batch_size=self.batch_size,
                verbose=self.predict_verbose
            )
        return zhat_rf

    def compile(self, X=None, y=None, y_rf=None, **kwargs):
        """
        See :class:`.DLModelHandler`,
        :meth:`dl_model_handler.DLModelHandler.compile`,
        :class:`.DLModelCompiler`, and
        :meth:`dl_model_compiler.DLModelCompiler.compile`.
        """
        if self.compiler is None:
            self.compiler = DLModelCompiler(self.compilation_args)
        return self.compiler.compile(self, X=X, y=y, y_rf=y_rf, **kwargs)

    # ---  MODEL HANDLING TASKS  --- #
    # ------------------------------ #
    def build_callbacks(self):
        """
        See :meth:`dl_model_handler.DLModelHandler.build_callbacks`.
        """
        return DLCallbackBuilder().build(self)

    def update_paths(self, model_args):
        """
        Consider the current specification of model handling arguments to
        update the paths.
        """
        if self.path_manager is None:
            self.path_manager = DLPathManager()
        self.path_manager.update_paths(model_args, self.arch)

    # ---  UTIL METHODS  --- #
    # ---------------------- #
    def build_sequencer(self, X, y_rf, training):
        """
        Build/instantiate a sequencer from the given input data and
        specification.

        :param X: The input data.
        :param y_rf: The input reference values.
        :param training: Whether the sequencer must be built for a training
            context (True) or a predictive context (False).
        :return: The built sequencer.
        :rtype: :class:`.DLSequencer`
        """
        # TODO Rethink : Move implementation to DLSequencerBuilder
        # Handle case where no sequencer is required
        if self.sequencer_spec is None:
            return X, y_rf
        # Build sequencer
        seq_type = self.sequencer_spec['type']
        seq_type_low = seq_type.lower()
        if seq_type_low == 'dlsequencer':
            return DLSequencer(
                X,
                y_rf,
                self.batch_size,
                arch=self.arch,
                augmentor=self.sequencer_spec.get('augmentor', None),
                random_shuffle_indices=self.sequencer_spec.get(
                    'random_shuffle_indices', False
                ),
                training=training
            ), None
        elif seq_type_low == 'dlsparseshadowsequencer':
            return DLSparseShadowSequencer(
                X,
                y_rf,
                self.batch_size,
                arch=self.arch,
                random_shuffle_indices=self.sequencer_spec.get(
                    'random_shuffle_indices', False
                ),
                training=training
            ), None
        else:
            raise DeepLearningException(
                'SimpleDLModelHandler did not expect the sequencer '
                f'"{seq_type}" so it could not be built.'
            )

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def __getstate__(self):
        """
        Method to be called when saving the serialized simple deep learning
        model handler.

        :return: The state's dictionary of the object
        :rtype: dict
        """
        # Obtain from parent
        state = super().__getstate__()
        # Update
        self.path_manager.add_to_state(state)
        state['out_prefix'] = self.out_prefix
        state['training_epochs'] = self.training_epochs
        state['batch_size'] = self.batch_size
        state['history'] = copy.copy(self.history)
        if state.get('history', None) is not None:
            state['history'].model = None  # Do not serialize keras/tf model
        state['checkpoint_monitor'] = self.checkpoint_monitor
        state['learning_rate_on_plateau'] = self.learning_rate_on_plateau
        state['early_stopping'] = self.early_stopping
        state['compilation_args'] = self.compilation_args
        state['sequencer_spec'] = self.sequencer_spec
        state['fit_verbose'] = self.fit_verbose
        state['predict_verbose'] = self.predict_verbose
        state['prediction_reducer'] = self.prediction_reducer
        # Return Simple DL Model Handler state (for serialization)
        return state

    def __setstate__(self, state):
        """
        Method to be called when loading and deserializing a previously
        serialized simple deep learning model handler.

        :param state: The state's dictionary of the saved simple deep learning
            model handler.
        :return: Nothing, but modifies the internal state of the object.
        """
        # Call parent
        super().__setstate__(state)
        # Assign member attributes from state dictionary
        self.out_prefix = state['out_prefix']
        self.training_epochs = state['training_epochs']
        self.batch_size = state['batch_size']
        self.history = state['history']
        self.checkpoint_monitor = state['checkpoint_monitor']
        self.learning_rate_on_plateau = state['learning_rate_on_plateau']
        self.early_stopping = state['early_stopping']
        self.compilation_args = state['compilation_args']
        self.sequencer_spec = state.get(
            'sequencer_spec', state.get('training_sequencer', None)
        )
        self.fit_verbose = state['fit_verbose']
        self.predict_verbose = state['predict_verbose']
        self.prediction_reducer = state.get(
            'prediction_reducer', PredictionReducer()
        )
        self.skip_fit_on_zero_epochs = VL3DCFG['MODEL'][
            'SimpleDLModelHandler'
        ].get('skip_fit_on_zero_epochs', False)
        self.path_manager = DLPathManager()
        self.path_manager.get_from_state(state)