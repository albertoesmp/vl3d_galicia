# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
from src.main.vl3d_exception import VL3DException
from src.utils.dict_utils import DictUtils
import numpy as np


# ---   EXCEPTIONS   --- #
# ---------------------- #
class PointTransformerException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to point transformation components.
    See :class:`.VL3DException`
    """
    def __init__(self, message=''):
        # Call parent VL3DException
        super().__init__(message)


# ---   CLASS   --- #
# ----------------- #
class PointTransformer:
    """
    :author: Alberto M. Esmoris Pena

    Class for point transformation operations.

    :ivar fnames: The names of the features involved in the transformation
        (by default None, which will lead to considering all the features
        in the point cloud).
    :vartype fnames: None or list of str
    :ivar _fnames: A cache with the names of the features involved in the
        CURRENT transformation. It can be different to ``fnames``. For example,
        when ``fnames`` is None, ``_fnames`` will contain the name for each
        feature in the point cloud. It must not be used outside the scope of
        an ongoing transformation (consistency is not guaranteed).
    :vartype _fnames: :class:`np.ndarray` of str
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_ptransf_args(spec):
        # Initialize
        kwargs = {
            'fnames': spec.get('fnames', None)
        }
        # Delete keys with None Value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a PointTransformer.

        :param kwargs: The attributes for the PointTransformer.
        """
        # Fundamental initialization of any point transformer
        self.fnames = kwargs.get('fnames', None)
        # Cached attributes (can be updated on different calls)
        self._fnames = None

    # ---  POINT TRANSFORM METHODS  --- #
    # --------------------------------- #
    @abstractmethod
    def transform(
        self, X=None, F=None, y=None, yhat=None, out_prefix=None, **kwargs
    ):
        """
        The fundamental transformation logic defining the point transformer.

        :param X: The structure space matrix of the point cloud to be
            transformed.
        :type X: :class:`np.ndarray` or None
        :param F: The feature space matrix of the point cloud to be
            transformed.
        :type F: :class:`np.ndarray` or None
        :param y: The references (if any) of the point cloud to be transformed.
        :type y: :class:`np.ndarray` or None
        :param yhat: The predictions (if any) on the point cloud to be
            transformed.
        :type yhat: :class:`np.ndarray` or None
        :param out_prefix: The output prefix (OPTIONAL). It might be used to
            particularize output paths, if any.
        :type out_prefix: str or None
        :param kwargs: Further key-word arguments defining the transformation.
        :type kwargs: dict
        """
        pass

    def transform_pcloud(self, pcloud, out_prefix=None, **kwargs):
        """
        Apply the transform method to a point cloud.

        See :meth:`point_transformer.PointTransformer.transform`.

        :param pcloud: The point cloud to be transformed.
        :type pcloud: :class:`.PointCloud`
        :param out_prefix: The output prefix (OPTIONAL). It might be used to
            particularize output paths, if any.
        :type out_prefix: str or None
        :param kwargs: Further key-word arguments defining the transformation.
        :type kwargs: dict
        :return: Whatever the transform method of the non-abstract derived
                class returns. Thus, in general, this method will need to be
                overridden to properly return a point cloud
                (see :class:`.PointCloud`).
        """
        # Get structure and feature spaces
        X = pcloud.get_coordinates_matrix()
        fnames = getattr(self, 'fnames', kwargs.get('fnames', None))
        if fnames is None:
            fnames = pcloud.get_features_names()
        self._fnames = np.array(fnames)
        F = pcloud.get_features_matrix(fnames)
        # Get classes vector as reference, if any
        y = pcloud.get_classes_vector() if pcloud.has_classes() else None
        # Get predictions, if any
        yhat = None
        if pcloud.has_predictions:
            yhat = pcloud.get_predictions_vector()
        # Return transformed point cloud
        pcloud.proxy_dump()
        return self.transform(
            X=X, F=F, y=y, yhat=yhat, out_prefix=out_prefix, **kwargs
        )
