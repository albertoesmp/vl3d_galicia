# ---   IMPORTS   --- #
# ------------------- #
from curses import has_key

from src.utils.ptransf.receptive_field import ReceptiveField
from src.main.vl3d_exception import VL3DException
import pyvl3dpp as vl3dpp
import tensorflow as tf
from tensorflow.lookup import StaticHashTable, KeyValueTensorInitializer

import numpy as np


# ---   CLASS   --- #
# ----------------- #
class ReceptiveFieldHierarchicalSG(ReceptiveField):
    r"""
    :author: Alberto M. Esmoris Pena

    Class representing a hierarchical receptive field based on sparse grids.

    A hierarchical receptive field is a special type of receptive field because
    it is composed of many receptive fields organized in a hierarchical manner.

    See :class:`.ReceptiveField` and :class:`.HierarchicalSGPreProcessorPP`.

    :ivar size: The size of the cell. Note that 3D cells are voxels with the
        same length for each edge.
    :vartype size: float
    :ivar w: The size of the submanifold convolutional window. It is given as
        a half size in terms of number of cells. For example, :math:`w_t`
        means that the submanifold convolutional window at depth :math:`t` will
        consider the active cell in the center, :math:`w_t` cells backward, and
        :math:`w_t` cells forward. If the dimensionality of the space is
        :math:`n_x`, the number of cells in a submanifold convolutional
        window will be given by :math:`(2w_t + 1)^{n_x}`. Note that any
        submanifold convolution has a stride or step size of one.
    :vartype w: :class:`np.ndarray` of int
    :ivar wD: The size of the downsampling convolutional window. It is given as
        a size in terms of number of cells. For example, :math:`w^D_t` means
        that the downsampling convolutional window at depth :math:`t` will
        consist of :math:`(w^D_t)^{n_x}` cells for a :math:`n_x`-dimensional
        space.
    :vartype wD: :class:`np.ndarray` of int
    :ivar sD: The step size or stride for the downsampling convolutional window.
        The downsampling window will move :math:`s^D_t` cells along the
        :math:`t` sparse grid to generate the cells of the :math:`t+1`-th sparse
        grid in the hierarchy.
    :vartype sD: :class:`np.ndarray` of int
    :ivar wU: The size of the upsampling convolutional window. It is given as
        a size in terms of number of cells. For example, math:`w^U_t` means
        that the upsampling convolutional window at depth :math:`t` will
        consist of :math:`(w^U_t)^{n_x}` for a :math:`n_x`-dimensional
        space.
    :vartype wU: :class:`np.ndarray` of int
    :ivar sU: The step size or stride for the upsampling convolutional window.
        The upsampling window will move :math:`s^U_t` cells along the
        :math:`t+1` sparse grid to generate the cells of the :math:`t`-th sparse
        grid in the hierarchy.
    :vartype sU: :class:`np.ndarray` of int
    :ivar h: A list of 2-tuples, each tuple represents the :math:`h_t` map at
        depth :math:`t` by storing the vector of keys (first element of the
        tuple) and the vector of values (second element of the tuple).
        The :math:`h_t(i)` value can be understood as the sequential index
        representing the active cell that corresponds to the cell index
        :math:`i` in the sparse grid at depth :math:`t`.
    :vartype h: list of tuple of :class:`np.ndarray`
    :ivar hD: The downsampling indexing. The :math:`h^D_{ti}` value can be
        understood as the index of the min vertex (e.g., the
        lower-left-backward for a 3D grid) for the convolutional window in the
        sparse grid at depth :math:`t` that generates the value for the
        :math:`i`-th active cell in the sparse grid at depth :math:`t+1`.
    :vartype hD: (list or :class:`np.ndarray`) of int
    :ivar hU: The upsampling indexing. The :math:`h^U_{ti}` value can be
        understood as the index of the min vertex (e.g., the
        lower-left-backward for a 3D grid) for the convolutional window in the
        sparse grid at depth :math:`t+1` that generates the value for the
        :math:`i`-th active cell in the sparse grid at depth :math:`t`.
    :vartype hU: (list or :class:`np.ndarray`) of int
    :ivar n: The axis-wise number of partitions for each sparse grid of the
        hierarchy.
    :vartype n: :class:`np.ndarray` of int
    :ivar A: The min point of the sparse grid at the first level of the
        hierarchy.
    :vartype A: :class:`np.ndarray`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a hierarchical SG receptive field object.

        :param kwargs: The key-word specification to instantiate the
            ReceptiveFieldHierarchicalSG.

        :Keyword Arguments:
            *   *size* (``float``) --
                The size of the cell (note that 3D cells are voxels with the
                same length for each edge.)
            *   *w* (``list of int``) --
                The size of the submanifold convolutional window. Note that
                this size is interpreted differently from the others. It
                applies twice, one for each direction along each axis.
            *   *wD* (``list of int``) --
                The size of the downsampling convolutional window.
            *   *sD* (``list of int``) --
                The step size or stride for the downsampling convolutions.
            *   *wU* (``list of int``) --
                The size of the upsampling convolutional window.
            *   *sU* (``list of int``) --
                The step size or stride for the upsampling convolutions.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.size = kwargs.get('cell_size', 1.0)
        self.w = kwargs.get('submanifold_window', [1, 1, 1, 1, 1])
        if not isinstance(self.w, np.ndarray):
            self.w = np.array(self.w, dtype=np.int32)
        self.wD = kwargs.get('downsampling_window', [2, 2, 2, 2])
        if not isinstance(self.wD, np.ndarray):
            self.wD = np.array(self.wD, dtype=np.int32)
        self.sD = kwargs.get('downsampling_stride', [2, 2, 2, 2])
        if not isinstance(self.sD, np.ndarray):
            self.sD = np.array(self.sD, dtype=np.int32)
        self.wU = kwargs.get('upsampling_window', [2, 2, 2, 2])
        if not isinstance(self.wU, np.ndarray):
            self.wU = np.array(self.wU, dtype=np.int32)
        self.sU = kwargs.get('upsampling_stride', [2, 2, 2, 2])
        if not isinstance(self.sU, np.ndarray):
            self.sU = np.array(self.sU, dtype=np.int32)
        # Validate attributes
        if self.size is None or self.size <= 0:
            raise VL3DException(
                'ReceptiveFieldHierarchicalSG needs (size > 0) to be '
                f'instantiated, but (size = {self.size}) was given.'
            )
        if len(self.w) < 1:
            raise VL3DException(
                'ReceptiveFieldHierarchicalSG cannot be instantiated without '
                'submanifold windows.'
            )
        if len(self.w) != (len(self.wD)+1):
            raise VL3DException(
                'ReceptiveFieldHierarchicalSG cannot be instantiated with '
                f'{len(self.w)} submanifold windows and '
                f'{len(self.wD)} downsampling windows. '
                'The number of submanifold windows must be the number of '
                'downsampling windows plus one.'
            )
        if len(self.wD) != len(self.sD):
            raise VL3DException(
                'ReceptiveFieldHierarchicalSG cannot be instantiated with '
                f'{len(self.wD)} downsampling windows but '
                f'{len(self.sD)} downsampling strides. '
                'The number of downsampling windows must be equal to the '
                'number of downsampling strides.'
            )
        if len(self.wD) != len(self.wU):
            raise VL3DException(
                'ReceptiveFieldHierarchicalSG cannot be instantiated with '
                f'{len(self.wD)} downsampling windows but '
                f'{len(self.wU)} upsampling windows. '
                'The number of downsampling windows must be equal to the '
                'number of upsampling windows.'
            )
        if len(self.wU) != len(self.sU):
            raise VL3DException(
                'ReceptiveFieldHierarchicalSG cannot be instantiated with '
                f'{len(self.wU)} upsampling windows but '
                f'{len(self.sU)} upsampling strides. '
                'The number of upsampling windows must be equal to the '
                'number of upsampling strides.'
            )
        # Attributes computed during fit
        self.h = [  # The static hash tables indexing active cells sequentially
            None for i in range(len(self.w))
        ]
        self.hD = [  # The downsampling indexing
            None for i in range(len(self.wD))
        ]
        self.hU = [  # The upsampling indexing
            None for i in range(len(self.wU))
        ]
        self.n = None  # Axis-wise number of partitions
        self.A = None  # Min point of the grid

    # ---  RECEPTIVE FIELD METHODS  --- #
    # --------------------------------- #
    def fit(self, X, x=None, structure_float_type=np.float64, id=None):
        """
        Fit the receptive field to represent the given points by constructing
        a hierarchy of sparse grids such that the first one has as active cells
        those in which there is at least one point from the input structure
        space.

        :param X: The input structure space matrix of :math:`m` points in a
            :math:`n_x`-dimensional space (for now, typically :math:`n_x=3`).
        :param x: The center point used to define the origin of the receptive
            field.
            Note that this parameter is ignored by the hierarchical sparse grid
            because it does not explicitly generate structure spaces nor
            centers the sparse grids in the hierarchy.
        :param x: Not used by
        :param structure_float_type: The decimal type for the structure space.
            Note that this parameter is ignored by the hierarchical sparse grid
            because it does not explicitly generate structure spaces.
        :param id: See :meth:`.ReceptiveField.fit`.
        :return: The fitted receptive field itself (for fluent programming).
        :rtype: :class:`.ReceptiveFieldHierarchicalSG`
        """
        # Validate input
        if X is None:
            raise ValueError(
                'ReceptiveFieldHierarchicalGS cannot fit without input '
                'points X.'
            )
        # Prepare the C++ function that must be called
        Xdtype = X.dtype
        cpp_f = None
        if Xdtype == np.float32:
            cpp_f = vl3dpp.rf_dl_sg_fit_fs32
        else:
            cpp_f = vl3dpp.rf_dl_sg_fit_ds32
        # Compute C++ function
        h, hD, hU, n, A = cpp_f(
            X,
            self.size,
            self.w,
            self.wD,
            self.wU,
            self.sD,
            self.sU,
            1
        )
        # Transform h dictionaries to TensorFlow static hash tables
        h = [
            [
                np.array(list(ht.keys()), dtype=np.int32),
                np.array(list(ht.values()), dtype=np.int32)+1
            ]
            for ht in h
        ]
        # Transform downsampling and upsampling maps to contain 1d vectors
        hD = [hDt.flatten() for hDt in hD]
        hU = [hUt.flatten() for hUt in hU]
        # Transform A to be a 1D vector
        A = A.flatten()
        # Assign output
        self.h = h
        self.hU = hU
        self.hD = hD
        self.n = np.array(n).squeeze()
        self.A = A.flatten()
        # Return self for fluent programming
        return self

    # ---  HIERARCHICAL SG RECEPTIVE FIELD METHODS  --- #
    # ------------------------------------------------- #
    def get_submanifold_maps(self):
        """
        Obtain the submanifold maps from the cell index (key, domain)
        to its corresponding sequential active cell index (value, codomain).

        :return: The submanifold maps.
        :rtype: list of :class:`tf.lookup.StaticHashTable`
        """
        return self.h

    def get_downsampling_vectors(self):
        """
        Obtain the downsampling vector for each depth connecting the min
        vertex of the convolutional window with its corresponding cell in the
        downsampled sparse grid.

        :return: The downsampling vectors.
        :rtype: list of :class:`np.ndarray` of int
        """
        return self.hD

    def get_upsampling_vectors(self):
        """
        Obtain the upsampling vector for each depth connecting the min vertex
        of the convolutional window with its corresponding cell in the
        upsampled sparse grid.

        :return: The upsampling vectors.
        :rtype: list of :class:`np.ndarray` of int
        """
        return self.hU

    def get_num_partitions(self):
        """
        Obtain the number of axis-wise partitions for each sparse grid in the
        hierarchy.

        :return: The number of axis-wise partitions for each sparse grid in the
            hierarchy.
        :rtype: :class:`np.ndarray` of int
        """
        return self.n

    def get_min_point(self):
        """
        Obtain the min point/vertex of the first sparse grid in the hierarchy.

        :return: The min vertex of the first sparse grid.
        :rtype: :class:`np.ndarray`
        """
        return self.A

    def get_max_depth(self):
        r"""
        Obtain the max depth of the hierarchy.

        :return: The max depth of the hierarchy.
        :rtype: int
        """
        return len(self.w)

    def get_submanifold_windows(self):
        """
        Obtain the submanifold convolutional window.

        :return: The submanifold convolutional window.
        :rtype: int
        """
        return self.w

    def get_downsampling_windows(self):
        """
        Obtain the downsampling convolutional window.

        :return: The downsampling convolutional window.
        :rtype: int
        """
        return self.wD

    def get_upsampling_windows(self):
        """
        Obtain the upsampling convolutional window.

        :return: The upsampling convolutional window.
        :rtype: int
        """
        return self.wU

    # ---   UTIL METHODS  --- #
    # ----------------------- #
    def compute_active_centroids(self, t):
        r"""
        Compute the centroid for each active cell of the sparse grid at depth
        :math:`t \in \mathbb{Z}_{\geq 0}` the hierarchy.

        This method assumes a 3D structure space so each centroid can be
        computed from the index of the active cell :math:`k` such that:

        .. math::

            \left\{\begin{array}{ll}
                x(k) =& \left[(k \mod n_z) + \dfrac{1}{2}\right] s + A_x  \\
                y(k) =& \left[
                    \left(\left\lfloor\dfrac{k}{n_z}\right\rfloor \mod n_y\right)
                + \dfrac{1}{2} \right] s + A_y \\
                z(k) =& \left[
                    \left\lfloor\dfrac{k}{n_y n_z}\right\rfloor
                + \dfrac{1}{2} \right] s + A_z
            \end{array}\right.

        Where :math:`n_x, n_y, n_z` represent the number of partitions along the
        :math:`x, y, z` axis for the sparse grid at depth :math:`t` and
        :math:`A = (A_x, A_y, A_z)` represent the min vertex of the first
        sparse grid in the hierarchy.

        **NOTE** that the centroids of sparse grids after the first depth will
        not respect the original scale.


        :return: Each cell represented as a centroid.
        :rtype: :class:`np.ndarray`
        """
        h = self.h[t]  # Keys are indices of active cells in first sparse grid
        n = self.n[:, t]  # Number of axis-wise partitions in first sparse grid
        nynz = n[2]*n[1]
        X = []  # Output as list
        # Compute centroids
        for k, v in zip(h[0], h[1]):
            zi = k % n[2]
            yi = (k // n[2]) % n[1]
            xi = k // nynz
            X.append([
                (xi+0.5)*self.size + self.A[0],
                (yi+0.5)*self.size + self.A[1],
                (zi+0.5)*self.size + self.A[2]
            ])
        # Return output as numpy array
        return np.array(X)

    def get_submanifold_map_as_dict(self, t, without_ground=False):
        """
        Obtain the submanifold map :math:`h_t` at depth :math:`t` as a python
        dictionary.

        :param t: The depth of the submanifold map that must be obtained.
        :type t: int
        :param without_ground: Whether the values (sequential indices of
            active cells) must be corrected to avoid accounting for the ground
            value (True) or not (False). The ground value is a convenience
            so the HSG receptive field can be used inside a neural network.
            It assumes the feature space matrix is expanded with a new row at
            the beginning that contains the ground value (0). However, the
            original submanifold map is not aware of this. If without_ground is
            True then this method will revert the returned submanifold map
            (as dictionary) so it does not consider the ground value.
        :type without_ground: bool
        :return: The submanifold map at depth :math:`t` as a dictionary.
        :rtype: dict
        """
        kt, vt = self.h[t]
        if without_ground:
            return dict(zip(kt, vt-1))
        return dict(zip(kt, vt))
