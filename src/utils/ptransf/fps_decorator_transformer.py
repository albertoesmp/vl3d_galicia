# ---   IMPORTS   --- #
# ------------------- #
from src.main.vl3d_exception import VL3DException
from src.utils.ptransf.receptive_field_fps import ReceptiveFieldFPS
from src.utils.str_utils import StrUtils
from scipy.spatial.kdtree import KDTree as KDT
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class FPSDecoratorTransformer:
    r"""
    :author: Alberto M. Esmoris Pena

    Class representing an FPS transformer that can be used to decorate
    different components of the framework.
    """
    # TODO Rethink : Doc ivars

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        r"""
        Initialize/instantiate a receptive field object.

        :param kwargs: The key-word specification to instantiate the
            FPSDecoratorTransformer.

        :Keyword Arguments:
            *   *num_points* (``int`` or ``str``) --
                The number of points :math:`R` the input points must be reduced
                to.
                In other words, for a given number of input points :math:`m_1`,
                the reduced number of points will be :math:`R`. For another,
                let us say different (i.e., :math:`m_1 \neq m_2`) number of
                points, the reduced number of points will also be :math:`R`.
                Alternatively, it can be given as a string. If so, it is
                understood as an expression that can be evaluated, where "m"
                is the number of input points. For instance: "m/2" means the
                FPS will consider half of the input points.
            * *num_encoding_neighbors* (``int``) --
                How many closest neighbors consider when doing reductions.
                For instance, for three encoding neighbors reducing a value
                means three points in the original domain will be considered
                to estimate the value in the representation domain.
                A value of zero means the representation will be generated
                but no neighborhood information connecting the points from
                the original domain to the representation domain will be
                computed.
            * *num_decoding_neighbors* (``int``) --
                How many closest neighbors consider when doing propagations.
                For instance, for three decoding neighbors propagating a value
                means three points in the representation domain will be
                considered to estimate the value in the original domain.
                A value of zero means the representation will be generated
                but no neighborhood information connecting the points from
                the representation domain back to the original domain will
                be computed.
            * *release_encoding_neighborhoods* (``bool``) --
                A flag to enable releasing the encoding neighborhoods after
                building the representation. It can be used to save memory when
                the topological information of the encoding process will not
                be used after generating the representation.
            * *fast* (``bool``) --
                A flag to enable the fast-computation mode. When True, a random
                uniform subsampling will be computed before the furthest point
                sampling so the latest is faster because it is not considering
                the entire input point cloud.
            * *threads* (``int``) --
                The number of threads to be used for parallel computations.
                Giving -1 means using all available cores.
            * *representation_report_path* (``str``) --
                Path where a point cloud with the points in the representation
                space will be exported.
        """
        # Assign attributes
        self.num_points = kwargs.get('num_points', None)
        self.num_encoding_neighbors = kwargs.get('num_encoding_neighbors', 0)
        self.num_decoding_neighbors = kwargs.get('num_decoding_neighbors', 0)
        self.release_encoding_neighborhoods = kwargs.get(
            'release_encoding_neighborhoods', False
        )
        self.fast = kwargs.get('fast', False)
        self.threads = kwargs.get('threads', -1)
        self.representation_report_path = kwargs.get(
            'representation_report_path', None
        )
        # Initialize attributes that will be generated when transforming
        self.N = None  # Neighborhood matrix with indices in the original space
        self.M = None  # Neighborhood matrix with indices in the representation
        # Validate number of points
        if self.num_points is None or self.num_points < 1:
            raise VL3DException(
                f'FPSDecoratorTransformer cannot work for {self.num_points} '
                'target number of points.'
            )
        # Validate number of encoding neighbors
        if self.num_encoding_neighbors is None or self.num_encoding_neighbors < 0:
            raise VL3DException(
                'FPSDecoratorTransformer cannot work for '
                f'{self.num_encoding_neighbors} encoding neighbors.'
            )
        # Validate number of decoding neighbors
        if self.num_decoding_neighbors is None or self.num_decoding_neighbors < 0:
            raise VL3DException(
                'FPSDecoratorTransformer cannot work for '
                f'{self.num_decoding_neighbors} decoding neighbors.'
            )

    # ---   FPS DECORATOR METHODS   --- #
    # --------------------------------- #
    def transform(self, X, F=None, y=None, out_prefix=None):
        """

        :param X: The structure space matrix (i.e., the matrix of coordinates).
        :param F: The feature space matrix (i.e., the matrix of features).
        :param y: The reference classes (i.e., the classification).
        :param out_prefix: The output prefix (OPTIONAL). It might be used by a
            report to particularize the output path.
        :return:
        """
        # TODO Rethink : Finish doxygen doc
        # Handle num points
        num_points = self.num_points
        if isinstance(num_points, str):  # If string, evaluate expression
            m = X.shape[0]  # Number of input points (for the expression)
            num_points = eval(StrUtils.to_numpy_expr(num_points))
        # Build support points
        num_points = eval(self.num_points) if isinstance(self.num_points, str)\
            else self.num_points
        rep_X = ReceptiveFieldFPS.compute_fps_on_3D_pcloud(
            X,
            num_points=num_points,
            fast=self.fast
        )
        # Compute encoding neighborhoods
        if self.num_encoding_neighbors > 0:
            kdt = KDT(X)
            self.N = kdt.query(
                rep_X,
                k=self.num_encoding_neighbors,
                workers=self.threads
            )[1]
        # Compute decoding neighborhoods
        if self.num_decoding_neighbors > 0:
            kdt = KDT(rep_X)
            self.M = kdt.query(
                X,
                k=self.num_decoding_neighbors,
                workers=self.threads
            )[1]
        # Release encoding neighborhoods if requested
        if self.release_encoding_neighborhoods:
            self.N = None
        # Encode features for representation
        rep_F = None if F is None else self.reduce(F)
        # Encode classes for representation
        rep_y = None if y is None else self.reduce(y)
        # Return
        return rep_X, rep_F, rep_y

    def transform_pcloud(self, pcloud, fnames=None, out_prefix=None):
        """

        :param pcloud: The point cloud to be transformed.
        :type pcloud: :class:`.PointCloud`
        :param fnames: A list with the names of the features to be considered.
            If an empty list is given, no features will be considered.
            If None is given, all features will be considered.
        :type fnames: None or list of str
        :param out_prefix: The output prefix (OPTIONAL). It might be used by a
            report to particularize the output path.
        :type out_prefix: None or str
        :return:
        """
        # TODO Rethink : Finish doxygen doc
        X = pcloud.get_coordinates_matrix()
        if fnames is not None and len(fnames) == 0:
            F = None
        else:
            if fnames is None:
                F = pcloud.get_features_matrix(pcloud.get_features_names())
            else:
                F = pcloud.get_features_matrix(fnames)
        y = pcloud.get_classes_vector() if pcloud.has_classes() else None
        return self.transform(X=X, F=F, y=y, out_prefix=out_prefix)

    def propagate(self, rep_x):
        # TODO Rethink : Doxygen doc (AKA decode)
        # Nearest neighbor reduction
        if self.num_decoding_neighbors == 1:
            return rep_x[self.M]
        # Mean of the neighborhood reduction for 1 feature
        if len(rep_x.shape) == 1:
            return np.mean(rep_x[self.M], axis=1)
        # Mean of the neighborhood reduction for >1 features
        return np.mean(rep_x[self.M].T, axis=1).T
        # TODO Rethink : Implement

    def reduce(self, x):
        # TODO Rethink : Doxygen doc (AKA encode)
        # Nearest neighbor reduction
        if self.num_encoding_neighbors == 1:
            return x[self.N]
        # Mean of the neighborhood reduction for 1 feature
        if len(x.shape) == 1:
            return np.mean(x[self.N], axis=1)
        # Mean of the neighborhood reduction for >1 features
        return np.mean(x[self.N].T, axis=1).T
