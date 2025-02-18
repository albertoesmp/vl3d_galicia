# ---   CLASS   --- #
# ----------------- #
class DLPathManager:
    """
    Class to handle the paths for deep learning models.

    :ivar summary_report_path: Path to the file where the summary report
        will be written, i.e., the report that summarizes the model's
        architecture.
    :vartype summary_report_path: str
    :ivar training_history_dir: Path to the directory where the training
        history plots and reports will be exported, i.e., information related
        to the  training along many epochs.
    :vartype training_history_dir: path
    :ivar checkpoint_path: The path where the model's checkpoint will be
        exported. It is used to keep the best model when using the checkpoint
        callback strategy during training.
    :vartype checkpoint_path: str
    :ivar feat_struct_repr_dir: The path where the
        information relative to :class:`.FeaturesStructuringLayer` layers will
        be exported.
    :vartype feat_struct_repr_dir: str
    :ivar rbf_feat_extract_repr_dir: The path where the information relative to
        :class:`.RBFFeatExtractLayer` layers will be exported.
    :vartype rbf_feat_extract_repr_dir: str
    :ivar rbf_feat_processing_repr_dir: The path where the information relative
        to :class:`.RBFFeatProcessingLayer` layers will be exported.
    :vartype rbf_feat_processing_repr_dir: str
    :ivar kpconv_representation_dir: The path where the information relative
        to :class:`.KPConvLayer` layers will be exported.
    :vartype kpconv_representation_dir: str
    :ivar skpconv_representation_dir: The path where the information relative
        to :class:`.StridedKPConvLayer` layers will be exported.
    :vartype skpconv_representation_dir: str
    :ivar lkpconv_representation_dir: The path where the information relative
        to :class:`.LightKPConvLayer` layers will be exported.
    :vartype lkpconv_representation_dir: str
    :ivar slkpconv_representation_dir: The path where the information relative
        to :class:`.StridedLightKPConvLayer` layers will be exported.
    :vartype slkpconv_representation_dir: str
    """
    def __init__(self, **kwargs):
        """
        Initialize a DLPathManager.
        """
        model_handling = kwargs.get('model_handling', {})
        self.summary_report_path = kwargs.get(
            'summary_report_path',
            model_handling.get('summary_report_path', None)
        )
        self.training_history_dir = kwargs.get(
            'training_history_dir',
            model_handling.get('training_history_dir', None)
        )
        self.checkpoint_path = kwargs.get(
            'checkpoint_path',
            model_handling.get('checkpoint_path', None)
        )
        self.feat_struct_repr_dir = kwargs.get(
            'features_structuring_representation_dir',
            model_handling.get(
                'features_structuring_representation_dir',
                None
            )
        )
        self.rbf_feat_extract_repr_dir = kwargs.get(
            'rbf_feat_extract_repr_dir',
            model_handling.get('rbf_feat_extract_repr_dir', None)
        )
        self.rbf_feat_processing_repr_dir = kwargs.get(
            'rbf_feat_processing_repr_dir',
            model_handling.get('rbf_feat_processing_repr_dir', None)
        )
        self.kpconv_representation_dir = kwargs.get(
            'kpconv_representation_dir',
            model_handling.get('kpconv_representation_dir', None)
        )
        self.skpconv_representation_dir = kwargs.get(
            'skpconv_representation_dir',
            model_handling.get('skpconv_representation_dir', None)
        )
        self.lkpconv_representation_dir = kwargs.get(
            'lkpconv_representation_dir',
            model_handling.get('lkpconv_representation_dir', None)
        )
        self.slkpconv_representation_dir = kwargs.get(
            'slkpconv_representation_dir',
            model_handling.get('slkpconv_representation_dir', None)
        )

    # ---  PATH HANDLING METHODS  --- #
    # ------------------------------- #
    def update_paths(self, model_args, arch):
        """
        Consider the current specification of model handling arguments to
        update the paths.

        :param model_args: The model arguments from where the new paths must
            be taken.
        :type model_args: dict
        :param arch: The neural network architecture that must be updated, see
            :class:`.Architecture`.
        :type arch: :class:`.Architecture`
        """
        # Nothing to do if no specification is given
        if model_args is None:
            return
        # Update model paths
        model_handling = model_args.get('model_handling', None)
        if model_handling is not None:
            self.summary_report_path = model_handling.get(
                'summary_report_path', None
            )
            self.training_history_dir = model_handling.get(
                'training_history_dir', None
            )
            self.checkpoint_path = model_handling.get(
                'checkpoint_path', None
            )
            self.feat_struct_repr_dir = model_handling.get(
                'features_structuring_representation_dir',
                self.feat_struct_repr_dir
            )
            self.rbf_feat_extract_repr_dir = model_handling.get(
                'rbf_feat_extract_repr_dir',
                self.rbf_feat_extract_repr_dir
            )
            self.rbf_feat_processing_repr_dir = model_handling.get(
                'rbf_feat_processing_repr_dir',
                self.rbf_feat_processing_repr_dir
            )
            self.kpconv_representation_dir = model_handling.get(
                'kpconv_representation_dir', self.kpconv_representation_dir
            )
            self.skpconv_representation_dir = model_handling.get(
                'skpconv_representation_dir', self.skpconv_representation_dir
            )
            self.lkpconv_representation_dir = model_handling.get(
                'lkpconv_representation_dir', self.lkpconv_representation_dir
            )
            self.slkpconv_representation_dir = model_handling.get(
                'slkpconv_representation_dir', self.slkpconv_representation_dir
            )
        # Update architecture paths
        if arch is not None:
            arch.architecture_graph_path = model_args.get(
                'architecture_graph_path', None
            )
            # Update pre-processor paths
            pre_processor = None
            if arch.pre_runnable is not None:
                if hasattr(arch.pre_runnable, "pre_processor"):
                    pre_processor = arch.pre_runnable.pre_processor
                else:
                    pre_processor = arch.pre_runnable
            if pre_processor is not None:
                pre_processor.update_paths(model_args.get(
                    'pre_processing', None
                ))

    # ---  SERIALIZATION UTILS  --- #
    # ----------------------------- #
    def add_to_state(self, state):
        """
        Add handled paths to state dictionary.

        :param state: The state dictionary from a :meth:`__getstate__` method.
            See
            :meth:`simple_dl_model_handler.SimpleDLModelHandler.__getstate__`
            for an example.
        :type state: dict
        :return: Nothing at all.
        """
        state['summary_report_path'] = self.summary_report_path
        state['training_history_dir'] = self.training_history_dir
        state['feat_struct_repr_dir'] = self.feat_struct_repr_dir
        state['rbf_feat_extract_repr_dir'] = self.rbf_feat_extract_repr_dir
        state['rbf_feat_processing_repr_dir'] = \
            self.rbf_feat_processing_repr_dir
        state['kpconv_representation_dir'] = self.kpconv_representation_dir
        state['skpconv_representation_dir'] = self.skpconv_representation_dir
        state['lkpconv_representation_dir'] = self.lkpconv_representation_dir
        state['slkpconv_representation_dir'] = self.slkpconv_representation_dir
        state['checkpoint_path'] = self.checkpoint_path

    def get_from_state(self, state):
        """
        Get paths to handle from state dictionary.

        :param state: The state dictionary from a :meth:`__setstate__` method.
            See
            :meth:`simple_dl_model_handler.SimpleDLModelHandler.__setstate__`
            for an example.
        :type state: dict
        :return: Nothing at all.
        """
        self.summary_report_path = state['summary_report_path']
        self.training_history_dir = state['training_history_dir']
        self.feat_struct_repr_dir = state.get('feat_struct_repr_dir', None)
        self.rbf_feat_extract_repr_dir = state.get(
            'rbf_feat_extract_repr_dir', None
        )
        self.rbf_feat_processing_repr_dir = state.get(
            'rbf_feat_processing_repr_dir', None
        )
        self.kpconv_representation_dir = state.get(
            'kpconv_representation_dir', None
        )
        self.skpconv_representation_dir = state.get(
            'skpconv_representation_dir', None
        )
        self.lkpconv_representation_dir = state.get(
            'lkpconv_representation_dir', None
        )
        self.slkpconv_representation_dir = state.get(
            'slkpconv_representation_dir', None
        )
        self.checkpoint_path = state['checkpoint_path']
