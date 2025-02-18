# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class ReceptiveField:
    r"""
    :author: Alberto M. Esmoris Pena

    Interface representing a receptive field. Any class must realize this
    interface if it aims to provide receptive field-like transformations.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        pass

    # ---   RECEPTIVE FIELD METHODS   --- #
    # ----------------------------------- #
    @abstractmethod
    def fit(self, X, x, structure_float_type=np.float64, id=None):
        """
        Fit the receptive field to represent the given points.

        :param X: The input matrix of m points in an n-dimensional space.
        :type X: :class:`np.ndarray`
        :param x: The center point used to define the origin of the receptive
            field.
        :type x: :class:`np.ndarray`
        :param structure_float_type: The decimal type for the structure space.
        :type structure_float_type: :class:`np.dtype`
        :param id: Either an integer or a string identifying the receptive
            field that is being fit. Alternatively it can be None. Note that
            this argument is meant to be used only for certain purposes, but it
            is not fundamental for the receptive field. An example can be
            identifying the current receptive field to name an output report
            (see :class:`.ReceptiveFieldOversamplingReport`,
            :meth:`.ReceptiveFieldFPS.fit`, and
            :meth:`.ReceptiveFieldFPS.compute_fps_on_3D_pcloud`).
        :type id: int or str
        :return: The fit receptive field itself (for fluent programming).
        :rtype: :class:`.ReceptiveField`
        """
        pass

    @abstractmethod
    def centroids_from_points(self, X):
        """
        Compute the centroids, i.e., the points representing the original
        point cloud in the receptive field space. The points do not need to be
        actual centroids. The name comes from typical rectangular-cells which
        are represented by computing the cell-wise centroid considering all
        points inside a given cell.

        :param X: The matrix of input points
        :type X: :class:`np.ndarray`
        :return: A matrix which rows are the points representing the centroids.
        :rtype: :class:`np.ndarray`
        """
        pass

    @abstractmethod
    def propagate_values(self, v):
        """
        Propagate given values, so they are associated to the points in the
        original space preceding the receptive field. In other words, the
        values v in the domain of the receptive field are transformed back
        to the original domain before the receptive field was computed.

        :param v: The values to be propagated from the receptive field back
            to the original domain.
        :type v: list
        :return: The output as a matrix when there are more than two values
            per point or the output as a vector when there is one value per
            point.
        :rtype: :class:`np.ndarray`
        """
        pass

    @abstractmethod
    def reduce_values(self, X, v, reduce_f=np.mean):
        """
        Reduce given values so there is one per centroid in the receptive field
        at most.

        :param X: The matrix of coordinates representing the centroids of
            the receptive field.
        :type X: :class:`np.ndarray`
        :param v: The vector of values to be reduced.
        :type v: :class:`np.ndarray`
        :param reduce_f: The function to reduce many values to a single one.
            By default, it is the mean.
        :type reduce_f: callable
        :return: The reduced vector.
        :rtype: :class:`np.ndarray`
        """
        pass

    # ---  MEMORY UTILS  --- #
    # ---------------------- #
    def canibalize(self, rf):
        """
        Take the values from given receptive field.

        :param rf: Receptive field whose attributes will be canibalized.
        """
        raise NotImplementedError(
            'ReceptiveField does not implement the canibalize method. '
            f'{self.__class__.__name__} must provide its own implementation.'
        )
