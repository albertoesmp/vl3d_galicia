# ---   IMPORTS   --- #
# ------------------- #
from src.utils.preds.prediction_reducer_factory import PredictionReducerFactory

# ---   CLASS   --- #
# ----------------- #
class DLPretrainedHandler:
    """
    Class to handle pretrained deep learning models.
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        pass

    # ---  PRETRAINED MODEL METHODS  --- #
    # ---------------------------------- #
    def overwrite_pretrained_model(self, mh, spec):
        """
        Assist the :meth:`model.Model.overwrite_pretrained_model` method for
        deep learning models.

        See :meth:`dl_model_handler.DLModelHandler.overwrite_pretrained_model`.

        :param mh: The model handler to be updated.
        :type mh: :class:`.DLModelHandler`
        :param spec: The key-word specification containing the model's
            arguments.
        :type spec: dict
        :return: Nothing at all.
        """
        spec_keys = spec.keys()
        # Overwrite baseline attributes of the deep learning model handler
        if 'model_handling' in spec_keys:
            spec_handling = spec['model_handling']
            spec_handling_keys = spec_handling.keys()
            if 'class_weight' in spec_handling_keys:
                mh.class_weight = spec_handling['class_weight']
            if 'class_names' in spec_handling_keys:
                mh.class_names = spec_handling['class_names']
            if 'summary_report_path' in spec_handling_keys:
                mh.path_manager.summary_report_path = \
                    spec_handling['summary_report_path']
            if 'training_history_dir' in spec_handling_keys:
                mh.path_manager.training_history_dir = \
                    spec_handling['training_history_dir']
            if 'checkpoint_path' in spec_handling_keys:
                mh.path_manager.checkpoint_path = \
                    spec_handling['checkpoint_path']
            if 'checkpoint_monitor' in spec_handling_keys:
                mh.checkpoint_monitor = spec_handling['checkpoint_monitor']
            if 'batch_size' in spec_handling_keys:
                mh.batch_size = spec_handling['batch_size']
            if 'training_epochs' in spec_handling_keys:
                mh.training_epochs = spec_handling['training_epochs']
            if 'learning_rate_on_plateau' in spec_handling_keys:
                mh.learning_rate_on_plateau = \
                    spec_handling['learning_rate_on_plateau']
            if 'early_stopping' in spec_handling_keys:
                mh.early_stopping = spec_handling['early_stopping']
            if 'training_sequencer' in spec_handling_keys:
                mh.training_sequencer = spec_handling['training_sequencer']
            if 'prediction_reducer' in spec_handling_keys:
                mh.prediction_reducer = PredictionReducerFactory.make_from_dict(
                    spec_handling['prediction_reducer']
                )
        # Overwrite compilation arguments
        if 'compilation_args' in spec_keys:
            mh.compilation_args = spec['compilation_args']
        # Overwrite the attributes of the model's architecture
        if mh.arch is not None:
            mh.arch.overwrite_pretrained_model(spec)
