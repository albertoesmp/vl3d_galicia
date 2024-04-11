# ---   IMPORTS   --- #
# ⨪------------------ #
from src.model.model import Model
from src.model.model import ModelException
import src.main.main_logger as LOGGING
from src.utils.dict_utils import DictUtils
from src.main.main_train import MainTrain
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade


# ---   CLASS   --- #
# ----------------- #
class FPSDecoratedModel(Model):
    """
    :author: Alberto M. Esmoris Pena

    Decorator for machine learning models that makes the decorated model work
    on a FPS-based representation of the point cloud.

    The FPS Decorated Model (:class:`.FPSDecoratedModel`)
    constructs a representation of the point cloud, then it calls
    the model on this representation and, when used for predicting, it
    propagates the predictions back to the original point cloud (the one from
    which the representation was built).

    :ivar decorated_model_spec: The specification of the decorated model.
    :vartype decorated_model_spec: dict
    :ivar decorated_model: The decorated model object.
    :vartype decorated_model: :class:`.Model`
    :ivar fps_decorator_spec: The specification of the FPS transformation
        defining the decorator.
    :vartype fps_decorator_spec: dict
    :ivar fps_decorator: The FPS decorator to be applied on input point clouds.
    :vartype fps_decorator: :class:`.FPSDecoratorTransformer`
    """
    # TODO Rethink : Update class from previous impl. try (RFMLModel)
    # TODO Rethink : Link with FPSDecoratorTransformer
    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_model_args(spec):
        """
        Extract the arguments to initialize/instantiate a FPSDecoratedModel
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a FPSDecoratedModel.
        """
        # Initialize from parent
        kwargs = Model.extract_model_args(spec)
        # Extract particular arguments for decorated machine learning models
        kwargs['decorated_model'] = spec.get('decorated_model', None)
        kwargs['fps_decorator'] = spec.get('fps_decorator', None)
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialization for any instance of type :class:`.FPSDecoratedModel`.
        """
        # Call parent init
        super().__init__(**kwargs)
        # Basic attributes of the FPSDecoratedModel
        self.decorated_model_spec = kwargs.get('decorated_model', None)
        self.fps_decorator_spec = kwargs.get('fps_decorator', None)
        # Validate decorated model as an egg
        if self.decorated_model_spec is None:
            LOGGING.LOGGER.error(
                'FPSDecoratedModel did not receive any model specification.'
            )
            raise ModelException(
                'FPSDecoratedModel did not receive any model specification.'
            )
        # Validate fps_decorator as an egg
        if self.fps_decorator_spec is None:
            LOGGING.LOGGER.error(
                'FPSDecoratedModel did not receive any decorator '
                'specification.'
            )
            raise ModelException(
                'FPSDecoratedModel did not receive any decorator '
                'specification.'
            )
        # Hatch validated model egg
        model_class = MainTrain.extract_model_class(self.decorated_model_spec)
        self.decorated_model = MainTrain.extract_pretrained_model(
            self.decorated_model_spec,
            expected_class=model_class
        )
        if self.decorated_model is None:  # Initialize model when no pretrained
            self.decorated_model = model_class(
                **model_class.extract_model_args(self.decorated_model_spec)
            )
        else:
            self.decorated_model.overwrite_pretrained_model(
                model_class.extract_model_args(self.decorated_model_spec)
            )
        # Hatch validated fps_decorator egg
        self.fps_decorator = FPSDecoratorTransformer(**self.fps_decorator_spec)
        # Warn about not recommended number of encoding neighbors
        nen = self.fps_decorator_spec.get('num_encoding_neighbors', None)
        if nen is None or nen != 1:
            LOGGING.LOGGER.warning(
                f'FPSDecoratedModel received {nen} encoding neighbors. '
                'Using a value different than one is not recommended as it '
                'could easily lead to unexpected behaviors.'
            )

    # ---   MODEL METHODS   --- #
    # ------------------------- #
    def train(self, pcloud):
        """
        Decorate the main training logic to work on the representation.
        See :class:`.Model` and :meth:`.Model.train`.
        """
        # Build representation from input point cloud
        fnames = self.get_fnames_recursively()
        rf_XF, rf_y = self.rf_pre_processor({
            'X': [
                pcloud.get_coordinates_matrix(),  # The coordinates
                pcloud.get_features_matrix(fnames)  # The features
            ],
            'y': pcloud.get_classes_vector()
        })
        # Build new point cloud from representation
        header = pcloud.get_header()
        rf_pcloud = PointCloudFactoryFacade.make_from_arrays(
            rf_XF[0], rf_XF[1], y=rf_y, header=header, scale=header.scales[0]
        )
        # Dump original point cloud to disk if space is needed
        pcloud.proxy_dump()
        # Train the model
        return self.decorated_model.train(rf_pcloud)

    def predict(self, pcloud, X=None):
        """
        Decorate the main predictive logic to work on the representation.
        See :class:`.Model` and :meth:`.Model.predict`.
        """
        # Build representation from input point cloud
        fnames = self.get_fnames_recursively()
        rf_XF = self.rf_pre_processor({
            'X': [
                pcloud.get_coordinates_matrix(),  # The coordinates
                pcloud.get_features_matrix(fnames)  # The features
            ]
        })
        # Build new point cloud from representation
        header = pcloud.get_header()
        rf_pcloud = PointCloudFactoryFacade.make_from_arrays(
            rf_XF[0], rf_XF[1], header=header, scale=header.scales[0]
        )
        # Dump original point cloud to disk if space is needed
        pcloud.proxy_dump()
        # Predict
        rf_yhat = self.decorated_model.predict(rf_pcloud)
        # Delete rf_pcloud and rf_XF features
        rf_pcloud = None
        rf_XF[1] = None
        # Propagate predictions to original point cloud and return
        return self.rf_post_processor({'X': rf_XF[0], 'z': rf_yhat})

    def prepare_model(self):
        """
        Prepare the decorated model.
        See :class:`.Model` and :meth:`.Model.prepare_model`.
        """
        self.decorated_model.prepare_model()

    def overwrite_pretrained_model(self, spec):
        """
        Overwrite the decorated pretrained model.
        See :class:`.Model` and :meth:`.Model.overwrite_pretrained_model`.
        """
        self.decorated_model.overwrite_pretrained_model(
            spec["decorated_model_spec"]
        )

    def get_input_from_pcloud(self, pcloud):
        """
        Get input from the decorated pretrained model.
        See :class:`.Model` and :meth:`.Model.get_input_from_pcloud`.
        """
        # TODO Rethink : Use representation here too?
        return self.decorated_model.get_input_from_pcloud(pcloud)

    def is_deep_learning_model(self):
        """
        Check whether the decorated model is a deep learning model.
        See :class:`.Model` and :meth:`.Model.is_deep_learning_model`.
        """
        return self.decorated_model.is_deep_learning_model()

    # ---   TRAINING METHODS   --- #
    # ---------------------------- #
    def training(self, X, y, info=True):
        """
        Use the training logic of the decorated model.
        See :class:`.Model` and :meth:`.Model.training`.
        """
        return self.decorated_model.training(X, y, info=info)

    def autoval(self, y, yhat, info=True):
        """
        Auto validation during training through decorated model.
        See :class:`.Model` and :meth:`.Model.autoval`.
        """
        return self.decorated_model.autoval(y, yhat, info=info)

    def train_base(self, pcloud):
        """
        Straightforward training through decorated model.
        See :class:`.Model` and :meth:`.Model.train_base`.
        """
        return self.decorated_model.train_base(pcloud)

    def train_autoval(self, pcloud):
        """
        Use autovalidation training strategy from decorated model.
        See :class:`.Model` and :meth:`.Model.train_autoval`.
        """
        return self.decorated_model.train_autoval(pcloud)

    def train_stratified_kfold(self, pcloud):
        """
        Use stratified k-fold training strategy from decorated model.
        See :class:`.Model` and :meth:`.Model.train_stratified_kfold`.
        """
        return self.decorated_model.train_stratified_kfold(pcloud)

    def on_training_finished(self, X, y):
        """
        Use on training finished callback from decorated model.
        See :class:`.Model` and :meth:`.Model.on_training_finished`.
        """
        self.decorated_model.on_training_finished(X, y)

    # ---  PREDICTION METHODS  --- #
    # ---------------------------- #
    def _predict(self, X, **kwargs):
        """
        Do the predictions through the decorated model.
        See :class:`.Model` and :meth:`.Model._predict`.
        """
        return self.decorated_model._predict(X, **kwargs)

    # ---  FPS DECORATED MODEL METHODS  --- #
    # ------------------------------------- #
    def get_fnames_recursively(self):
        """
        Find through any potential decoration graph until the deepest model
        is found, then consider its feature names.
        :return: The feature names of the deepest model in the decoration
            hierarchy.
        :rtype: list of str
        """
        # Find through all decorated models until the last one (deepest)
        submodel = self.decorated_model
        while hasattr(submodel, 'decorated_model'):
            submodel = submodel.decorated_model
        # Return the feature names of the deepest model
        return submodel.fnames
