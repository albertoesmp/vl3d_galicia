# ---   IMPORTS   --- #
# ------------------- #
from src.utils.ptransf.receptive_field import ReceptiveField
from src.report.receptive_field_oversampling_report import \
    ReceptiveFieldOversamplingReport
from scipy.spatial import KDTree as KDT
import numpy as np
import open3d


# ---   CLASS   --- #
# ----------------- #
class ReceptiveFieldFPS(ReceptiveField):
    r"""
    :author: Alberto M. Esmoris Pena

    Class representing a receptive field based on furthest point subsampling.

    See :class:`.ReceptiveField`, :class:`.ReceptiveFieldGS`, and
    :class:`.FurthestPointSubsamplingPreProcessor`.

    :ivar num_points: The number of points so each point cloud is subsampled
        to this number of points through FPS. Typically noted as :math:`R`.
    :vartype num_points: int
    :ivar num_encoding_neighbors: How many neighbors consider when propagating
        and reducing. Assume the number of encoding neighbors is
        :math:`m^* \in \mathbb{Z}_{\geq 0}`. For then, when reducing values
        from :math:`\pmb{X} \in \mathbb{R}^{m \times n}` (input point cloud)
        to :math:`\pmb{Y} \in \mathbb{R}^{R \times n}` (receptive field
        points), each reduced value in :math:`\pmb{Y}` will be obtained
        by reducing :math:`m^*` values in :math:`\pmb{X}`. Also, when
        propagating values from :math:`\pmb{Y} \in \mathbb{R}^{R \times n}` to
        :math:`\pmb{X} \in \mathbb{R}^{m \times n}`, each propagated value in
        :math:`\pmb{X}` will be obtained by reducing :math:`m^*` values from
        :math:`\pmb{Y}`.
    :vartype num_encoding_neighbors: int
    :ivar fast: Flag to control whether to use the fast mode or not. When
        running the FPS receptive field in fast mode, a random uniform sampling
        is computed before the furthest poit subsampling. While faster because
        it reduces the computational burden for the FPS, this approach is also
        less stable and might produce unexpected results.
    :vartype fast: bool
    :ivar N: The indexing matrix
        :math:`\pmb{N} \in \mathbb{Z}_{\geq 0}^{R \times m^*}`. Each row
        :math:`i` in this
        matrix represents the indices in :math:`\pmb{X}` that are associated
        to the point represented by the row :math:`i` in :math:`\pmb{Y}`.
    :vartype N: :class:`np.ndarray`
    :ivar M: The reverse indexing matrix
        :math:`\pmb{M} \in \mathbb{Z}_{\geq 0}^{m \times m^*}. Each row
        :math:`i` in this matrix represents the indices in :math:`\pmb{Y}` that
        are associated to the points represented by the row :math:`i` in
        :math:`\pmb{X}`.
    :vartype M: :class:`np.ndarray`
    :ivar x: The center point of the receptive field. It is assigned when
        calling :meth:`receptive_field_fps.ReceptiveFieldFPS.fit`.
    :vartype x: :class:`npn.ndarray`
    :ivar Y: The subsample representing the original input point cloud, i.e.,
        a matrix of coordinates in a :math:`n`-dimensional space such that
        :math:`\pmb{Y} \in \mathbb{R}^{R \times n}`.
    :vartype Y: :class:`np.ndarray`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        r"""
        Initialize/instantiate a receptive field object.

        :param kwargs: The key-word specification to instantiate the
            ReceptiveFieldFPS.

        :Keyword Arguments:
            *   *num_points* (``int``) --
                The number of points :math:`R` the input points must be reduced
                to.
                In other words, for a given number of input points :math:`m_1`,
                the reduced number of points will be :math:`R`. For another,
                let us say different (i.e., :math:`m_1 \neq m_2`) number of
                points, the reduced number of points will also be
                :math:`R`.
            * *num_encoding_neighbors* (``int``) --
                How many neighbors consider when doing propagations and
                reductions. For instance, for three encoding neighbors
                propagating a value means three points in the receptive
                field will be considered to estimate the value in the
                original domain. Analogously, reducing a value means three
                points in the original domain will be considered to encode
                the value in the receptive field.
            * *fast* (``bool``) --
                A flag to enable the fast-computation mode. When True, a random
                uniform subsampling will be computed before the furthest point
                sampling so the latest is faster because it is not considering
                the entire input point cloud.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.num_points = kwargs.get('num_points', 8000)
        self.num_encoding_neighbors = kwargs.get('num_encoding_neighbors', 3)
        self.fast = kwargs.get('fast', False)
        self.oversampling = kwargs.get(
            'receptive_field_oversampling',
            kwargs.get('oversampling', None)
        )
        self.N = None  # The indexing matrix will be created during fit
        self.M = None  # The reverse indexing matrix will be created during fit
        self.x = None  # The center point of the receptive field
        self.Y = None  # The centroids of the receptive field

    # ---   RECEPTIVE FIELD METHODS   --- #
    # ----------------------------------- #
    def fit(self, X, x, structure_float_type=np.float64, id=None):
        """
        Fit the receptive field to represent the given points by taking the
        subset of the furthest points, i.e., the subset of points that maximize
        the distances between points. Typically, the next point in a FPS
        iteration maximizes the distance with respect to the already
        considered points in a greedy scheme.

        :param X: The input matrix of m points in an n-dimensional space.
        :type X: :class:`np.ndarray`
        :param x: The center point used to define the origin of the receptive
            field.
        :type x: :class:`np.ndarray`
        :param structure_float_type: The decimal type for the structure space.
        :type structure_float_type: :class:`np.dtype`
        :param id: See :meth:`ReceptiveField.fit`.
        :return: The fitted receptive field itself (for fluent programming)
        :rtype: :class:`.ReceptiveFieldFPS`
        """
        # Validate input
        if x is None:
            raise ValueError(
                'ReceptiveFieldFPS cannot fit without an input center point x.'
            )
        if X is None:
            raise ValueError(
                'ReceptiveFieldFPS cannot fit without input points X.'
            )
        # Center and scale the input point cloud
        self.x = x
        X = self.center_and_scale(X)
        # Compute the FPS "centroids"
        self.Y = ReceptiveFieldFPS.compute_fps_on_3D_pcloud(
            X,
            fast=self.fast,
            num_points=self.num_points,
            structure_float_type=structure_float_type,
            oversampling=self.oversampling
        )
        # Find the indexing matrix N
        kdt = KDT(X)
        self.N = kdt.query(self.Y, k=self.num_encoding_neighbors)[1].astype(
            np.int32  # Cannot be uint32 because tensorflow demands signed
        )
        if len(self.N.shape) < 2:
            self.N = self.N.reshape(-1, 1)
        # Find the indexing matrix M
        kdt = KDT(self.Y)
        self.M = kdt.query(X, k=self.num_encoding_neighbors)[1].astype(
            np.int32  # Cannot be uint32 because tensorflow demands signed
        )
        if len(self.M.shape) < 2:
            self.M = self.M.reshape(-1, 1)
        # Return self for fluent programming
        return self

    def centroids_from_points(self, X):
        """
        The centroids of an FPS receptive field are said to be the subsampled
        points themselves.

        :param X: The matrix of input points (can be NONE, in fact, it is not
            used).
        :type X: :class:`np.ndarray` or None
        :return: A matrix which rows are the points representing the centroids.
        :rtype: :class:`np.ndarray`
        """
        return self.Y

    def propagate_values(self, v, reduce_strategy='mean', **kwargs):
        r"""
        Propagate :math:`R` values associated to
        :math:`\pmb{Y} \in \mathbb{R}^{R \times n}` to :math:`m`
        values associated to :math:`\pmb{X} \in \mathbb{R}^{m \times n}`
        through the indexing matrix
        :math:`\pmb{M} \in \mathbb{Z}_{\geq 0}^{m \times m^*}`.

        :param v: The :math:`R` values to be propagated.
        :type v: list
        :param reduce_strategy: The reduction strategy, either "mean" or
            "closest".
        :type reduce_strategy: str
        :return: The output as a matrix when there are more than two values per
            point or the output as a vector when there is one value per point.
        :rtype: :class:`np.ndarray`
        """
        return ReceptiveFieldFPS.do_propagate_values(
            self.M, v, reduce_strategy
        )

    @staticmethod
    def do_propagate_values(M, v, reduce_strategy):
        """
        See :meth:`ReceptiveFieldFPS.propagate_values`.
        """
        # Determine the dimensionality of each value (both scalar and vectors
        # can be propagated). All values must have the same dimensionality.
        try:
            val_dim = len(v[0])
        except Exception as ex:
            val_dim = 1
        # Prepare output matrix
        Ytype = v.dtype if isinstance(v, np.ndarray) else type(v[0])
        Y = np.full([len(M), val_dim], 0, dtype=Ytype)
        # Populate output matrix : Reduce by mean
        if reduce_strategy == 'mean':
            for i, Mi in enumerate(M):
                Y[i] = np.mean(v[Mi], axis=0)
        # Populate output matrix : Take from closest
        elif reduce_strategy == 'closest':
            for i, Mi in enumerate(M):
                Y[i] = v[Mi[0]]
        else:  # Unexpected reduce strategy
            raise ValueError(
                'The FPS receptive field received an unexpected '
                'reduce_strategy when propagating values.'
            )
        # Return output matrix (or vector if single-column)
        return Y if Y.shape[1] > 1 else Y.flatten()

    def reduce_values(self, X, v, reduce_f=np.mean):
        r"""
        Reduce :math:`m` values associated to
        :math:`\pmb{X} \in \mathbb{R}^{m \times n} to :math:`R` values
        associated to :math:`\pmb{Y} \in \mathbb{R}^{R \times n}` through the
        indexing matrix
        :math:`\pmb{N} \in \mathbb{Z}_{\geq 0}^{m \times m^*}`.

        :param X: The centroids representing the furthest point subsampling
            computed by the receptive field. It can be None since it is not
            used for an FPS receptive field.
        :type X: :class:`np.ndarray` or None
        :param v: The vector of values to reduce. The :math:`m` input
            components will be reduced to :math:`R` output components.
        :param reduce_f: The function to reduce many values to a single
            one. By default, it is mean.
        :type reduce_f: callable
        :return: The reduced vector.
        :rtype: :class:`np.ndarray`
        """
        # TODO Rethink : Shall X be removed from args?
        return ReceptiveFieldFPS.do_reduce_values(self.N, X, v, reduce_f)

    @staticmethod
    def do_reduce_values(N, X, v, reduce_f):
        """
        See :meth:`ReceptiveFieldFPS.reduce_values`.
        """
        # TODO Rethink : Shall X be removed from args?
        # Reduce
        v_reduced = np.zeros(len(N), dtype=v.dtype)
        for i, Ni in enumerate(N):
            v_reduced[i] = reduce_f(v[Ni])
        # Return
        return v_reduced

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    def center_and_scale(self, X):
        """
        Like :meth:`receptive_field_gs.ReceptiveFieldGS.center_and_scale` but
        without scaling, i.e., only centering.
        """
        return X - self.x

    def undo_center_and_scale(self, X):
        """
        Like :meth:`receptive_field_gs.ReceptiveFieldGS.undo_center_and_scale`
        but without scaling, i.e., only centering.
        """
        return X + self.x

    @staticmethod
    def compute_fps_on_3D_pcloud(
        X,
        num_points=None,
        fast=False,
        structure_float_type=np.float64,
        id=None,
        oversampling=None
    ):
        r"""
        Compute the furthest point sampling (FPS) algorithm on the point cloud
        represented by the input 3D matrix
        :math:`\pmb{X} \in \mathbb{R}^{m \times 3}`. The result is an output
        matrix :math:`\pmb{Y} \in \mathbb{R}^{R \times 3}`.

        :param X: The input 3D matrix (rows are points, columns dimensions).
        :type X: :class:`np.ndarray`
        :param num_points: The number of points :math:`R` selected through the
            furthest point sampling method.
        :type num_points: int
        :param fast: Whether to use a fast approximation of FPS (True) or
            the exact computation (False). The fast approximation is computed
            through uniform down sample. Alternatively, it can be (2) to use
            the turbo-fast mode (faster but purely stochastic). Note that
            turbo-fast can be slower than fast when only a few points (relative
            to the total) are selected, e.g., when selecting 5,000 or 10,000
            points from a point cloud of 80 millions.
        :type fast: bool or int
        :param structure_float_type: The decimal type for the structure space.
        :type structure_float_type: :class:`np.dtype`
        :param id: See :meth:`.ReceptiveField.fit`.
        :param oversampling: The dictionary governing the oversampling strategy
            for not populated enough receptive fields (OPTIONAL).
        :type oversampling: dict or None
        :return: The subsampled point cloud.
        :rtype: :class:`np.ndarray`
        """
        # Do the oversampling
        if X.shape[0] < num_points and oversampling is not None:
            # Apply oversampling, if requested and conditions are met
            return ReceptiveFieldFPS.oversample(
                X, num_points, **oversampling, id=id
            ).astype(structure_float_type)
        # Do the downsampling
        if fast == 2:  # Turbo-fast (see FPSDecoratorTransformer.transform)
            np.random.shuffle(X)
            return X[:num_points].astype(structure_float_type)
        # Prepare Open3D context
        o3d_cloud = open3d.geometry.PointCloud()
        o3d_cloud.points = open3d.utility.Vector3dVector(X)
        if fast:  # Just fast (uniform down sampling before exact FPS)
            step = X.shape[0] // num_points
            o3d_cloud = o3d_cloud.uniform_down_sample(step)
        # Do the exact furthest point sampling
        o3d_cloud = o3d_cloud.farthest_point_down_sample(num_points)
        # Raise error due to unreliable receptive field
        if len(o3d_cloud.points) != num_points:
            # Try an oversampling correction, if possible
            if oversampling is not None and len(o3d_cloud.points) < num_points:
                Y = np.asarray(o3d_cloud.points, dtype=structure_float_type)
                Y = ReceptiveFieldFPS.oversample(
                    Y, num_points, **oversampling, id=id
                )
                if Y.shape[0] == num_points:  # Check oversampling correctness
                    return Y  # Return oversampling correction, if correct
            # Otherwise (or if the correction failed) throw an exception
            raise ValueError(
                f'ReceptiveFieldFPS failed to sample {num_points}. Only '
                f'{len(o3d_cloud.points)} samples were taken for a given '
                f'input of {X.shape[0]} points.' + (
                    '' if len(o3d_cloud.points) >= X.shape[0] else
                    '\nFarthest point down sample might discard points under '
                    'some circumstances, e.g., repeated points.'
                )
            )
        return np.asarray(o3d_cloud.points, dtype=structure_float_type)

    @staticmethod
    def oversample(X, target_points, **kwargs):
        r"""
        Oversample the given structure space matrix
        :math:`\pmb{X} \in \mathbb{R}^{m \times n_x}` representing a point cloud
        with a :math:`n_x`-dimensional structure space. For a given target
        number of points :math:`m^* \in \mathbb{Z}_{>1}` The oversampling can
        be seen as a map
        :math:`\operatorname{o} : \mathbb{R}^{m \times n_x} \to \mathbb{R}^{m^* \times n_x}`
        .

        When adding new points, it is possible to have more candidates than
        needed extra points. In these cases, the tie is break by considering
        the best conditioned case. For example, when using the nearest
        strategy, those pairs of neighbors that are closer to each
        other will be selected first until :math:`m_* = m^*-m` points are added.

        :param X: The input structure space matrix
            :math:`\pmb{X} \in \mathbb{R}^{m \times n_x}`.
        :type X: :class:`np.ndarray`
        :param target_points: :math:`m^* \in \mathbb{Z}_{>1}`.
        :type target_points: int
        :param kwargs: The key-word specification governing the oversampling.
            It supports:

            -- ``min_points`` (:math:`m_*`, default 0)
                The minimum acceptable number of points. Input point clouds
                with :math:`m < m_*` points will raise an exception as they
                are considered unreliable.

            -- ``strategy`` (default "nearest")
                The oversampling strategy. It can be either
                ``"nearest"`` (for each point, the nearest neighbor distinct to
                itself is considered, a new point in the middle and equidistant
                to both neighbors is added), ``"knn"`` (the mean of the
                :math:`k`-nearest neighbors is added as a new point),
                ``"spherical"`` (the centroid of a spherical neighborhood is
                added as a new point), ``"gaussian_knn"`` (the
                :math:`k`-nearest neighbors are aggregated depending on their
                distance to the centroid of the neighborhood), and
                ``"spherical_radiation"`` (the points in the spherical
                neighborhood have a greater weight when computing the centroid
                depending on their distance to the center of the sphere, i.e.,
                the closest to the center, the greater the contribution).

            -- ``k`` (default 16)
                The number of :math:`k`-nearest neighbors for the knn and
                gaussian_knn strategies.

            -- ``radius`` (default 1)
                The radius for the spherical and spherical radiation
                strategies.

            -- ``nthreads`` (default 1)
                The number of threads to be used for parallel computations, if
                any. Default is one because the oversampling is typically
                computed inside an already parallelized region, think twice
                before changing this value.

            -- ``report_dir`` (default None)
                Path to the directory where the oversampled receptive fields
                will be exported as point clouds. If not given, not even a
                single point cloud will be written.

            -- ``ìd`` (default None)
                See :meth:`.ReceptiveField.fit`.

        :type kwargs: dict
        """
        # Extract variables from key-word specification
        min_points = kwargs.get('min_points', 0)
        strategy = kwargs.get('strategy', 'nearest')
        k = kwargs.get('k', 16)
        r = kwargs.get('radius', 1.0)
        nthreads = kwargs.get('nthreads', 1)
        oversampling_report_dir = kwargs.get('report_dir', None)
        # Check the min points threshold is satisfied, i.e., m >= m_*
        if X.shape[0] < min_points:
            raise ValueError(
                'ReceptiveFieldFPS failed to oversample a receptive field '
                f'because it has {X.shape[0]} points but at least {min_points} '
                'points are required for a reliable oversampling.'
            )
        K = target_points - X.shape[0]
        if K < 1:
            raise ValueError(
                'ReceptiveFieldFPS failed to oversample a receptive field '
                f'because it already has enough points {X.shape[0]} out of '
                f'{target_points} (cannot oversample {K} points).'
            )
        # Call corresponding oversampling method
        strategy_low = strategy.lower()
        if strategy_low == 'nearest':
            Y = ReceptiveFieldFPS.nearest_oversample(
                K, X, nthreads=nthreads
            )
        elif strategy_low == 'knn':
            Y = ReceptiveFieldFPS.knn_oversample(
                K, X, k, nthreads=nthreads
            )
        elif strategy_low == 'spherical':
            Y = ReceptiveFieldFPS.spherical_oversample(
                K, X, r, nthreads=nthreads
            )
        elif strategy_low == 'spherical_naive':
            Y = ReceptiveFieldFPS.naive_spherical_oversample(
                K, X, r, target_points, nthreads=nthreads
            )
        elif strategy_low == 'gaussian_knn':
            Y = ReceptiveFieldFPS.gaussian_knn_oversample(
                K, X, k, nthreads=nthreads
            )
        elif strategy_low == 'spherical_radiation':
            Y = ReceptiveFieldFPS.spherical_radiation_oversample(
                K, X, r, nthreads=nthreads
            )
        elif strategy_low == 'spherical_radiation_naive':
            Y = ReceptiveFieldFPS.naive_spherical_radiation_oversample(
                K, X, r, target_points, nthreads=nthreads
            )
        else:
            raise ValueError(
                'ReceptiveFieldFPS failed to oversample a receptive field '
                f'due to an unexpected strategy: "{strategy}"'
            )
        # Report oversampling, if requested
        if oversampling_report_dir is not None:
            ReceptiveFieldOversamplingReport(
                X=X, Y=Y, id=kwargs.get('id', None)
            ).to_file(oversampling_report_dir)
        # Return oversampled point cloud / neighborhood
        return Y

    # ---   OVERSAMPLING METHODS   --- #
    # -------------------------------- #
    @staticmethod
    def nearest_oversample(K, X, nthreads=1):
        r"""
        Let :math:`\pmb{X} \in \mathbb{R}^{m \times n_x}` be a structure
        space matrix. If :math:`m < m^*`, then :math:`K = m^*-m` points
        must be generated through nearest oversampling. This consists of
        finding the closest neighbor distinct to itself for each
        :math:`\pmb{x}_{i*}` and considering the mid-range from the
        :math:`K` closer pairs of neighbors.

        :param K: How many points must be sampled, i.e.,
            :math:`K = m^* - m`.
        :type K: int
        :param X: The structure space matrix.
        :type X: :class:`np.ndarray`
        """
        kdt = KDT(X)
        D, I = kdt.query(X, k=2, workers=nthreads)
        D, I = D[:, 1], I[:, 1]
        Knext = K-len(I)  # How many points oversample in the future
        Know = min(K, len(I))  # How many points oversample now
        S = np.argsort(D)[:Know]  # Select the Know closer pairs
        Y = (X[S]+X[I[S]]) / 2.0
        # Recursive oversampling
        if Knext > 0:
            return ReceptiveFieldFPS.nearest_oversample(
                Knext, np.vstack([X, Y]), nthreads=nthreads
            )
        # Straightforward return
        return np.vstack([X, Y])

    @staticmethod
    def knn_oversample(K, X, k, nthreads=1):
        r"""
        Let :math:`\pmb{X} \in \mathbb{R}^{m \times n_x}` be a structure
        space matrix. If :math:`m < m^*`, then :math:`K = m^*-m` points
        must be generated through k-nearest neighbors oversampling. This
        consists of finding the k-closest neighbors for each
        :math:`\pmb{x}_{i*}` and considering their centroid as a new point.
        In case more new points than needed are generated, those with the
        smallest distance between :math:`\pmb{x}_{i*}` and its furthest
        nearest neighbor will be prioritized.

        :param K: How many points must be sampled, i.e.,
            :math:`K = m^* - m`.
        :type K: int
        :param X: The structure space matrix.
        :type X: :class:`np.ndarray`
        :param k: The number of nearest neighbors to consider (:math:`k`)
        :type k: int
        """
        kdt = KDT(X)
        D, I = kdt.query(X, k=k, workers=nthreads)
        Knext = K-len(I)  # How many points oversample in the future
        Know = min(K, len(I))  # How many points oversample now
        S = np.argsort(D[:, -1])[:Know]  # Closest furthest pairs first
        Y = np.mean(X[I[S]], axis=1)  # Compute centroids
        # Recursive oversampling
        if Knext > 0:
            return ReceptiveFieldFPS.knn_oversample(
                Knext, np.vstack([X, Y]), k, nthreads=nthreads
            )
        # Straightforward return
        return np.vstack([X, Y])

    @staticmethod
    def spherical_oversample(K, X, r, nthreads=1):
        r"""
        Let :math:`\pmb{X} \in \mathbb{R}^{m \times n_x}` be a structure space
        matrix. If :math:`m < m^*`, then :math:`K = m^*-m` points must be
        generated through spherical oversampling. This consists of finding the
        spherical neighborhood centered on :math:`\pmb{x}_{i*}`, for each i-th
        point in the input structure space, and generating a new point by
        computing the centroid of the neighborhood. In case more new points
        than needed are generated, the furthest point sampling of the
        new points will be computed to select exactly :math:`m^*` points.

        :param K: How many points must be sampled, i.e.,
            :math:`K = m^* - m`.
        :type K: int
        :param X: The structure space matrix.
        :type X: :class:`np.ndarray`
        :param r: The radius for the spherical neighborhood :math:`r`.
        :type r: float
        """
        kdt = KDT(X)
        I = kdt.query_ball_tree(kdt, r)
        Y = np.unique([  # Compute centroids
            np.mean(X[Ii], axis=0) for Ii in I
        ], axis=0)
        Knext = K-Y.shape[0]  # How many points oversample in the future
        # Recursive oversampling
        if Knext > 0:
            return ReceptiveFieldFPS.spherical_oversample(
                Knext, np.vstack([X, Y]), r, nthreads=nthreads
            )
        # Reduce through FPS, if needed
        target_points = X.shape[0] + K
        if (X.shape[0] + Y.shape[0]) > target_points:
            o3d_cloud = open3d.geometry.PointCloud()
            o3d_cloud.points = open3d.utility.Vector3dVector(np.vstack([X, Y]))
            o3d_cloud = o3d_cloud.farthest_point_down_sample(target_points)
            if len(o3d_cloud.points) != target_points:
                raise ValueError(
                    'ReceptiveFieldFPS failed to reduce an spherical '
                    'oversample. Probably due to repeated points.'
                )
            return np.asarray(o3d_cloud.points)
        # Straightforward return
        return np.vstack([X, Y])

    @staticmethod
    def naive_spherical_oversample(K, X, r, target_points, nthreads=1):
        r"""
        Let :math:`\pmb{X} \in \mathbb{R}^{m \times n_x}` be a structure space
        matrix. If :math:`m < m^*`, then :math:`K = m^*-m` points must be
        generated through spherical oversampling. This consists of finding the
        spherical neighborhood centered on :math:`\pmb{x}_{i*}`, for each i-th
        point in the input structure space, and generating a new point by
        computing the centroid of the neighborhood. In case more new points
        than needed are generated, the first :math:`m^*` are selected in
        whatever order they are. Moreover, no uniqueness is forced on the
        centroids. That is why this method can be considered as the naive
        version of :meth:`.ReceptiveFieldFPS.spherical_oversample`.

        :param K: How many points must be sampled, i.e.,
            :math:`K = m^* - m`.
        :type K: int
        :param X: The structure space matrix.
        :type X: :class:`np.ndarray`
        :param r: The radius for the spherical neighborhood :math:`r`.
        :type r: float
        :param target_points: The target number of points.
        :type target_points: int
        """
        kdt = KDT(X)
        I = kdt.query_ball_tree(kdt, r)
        Y = np.array([  # Compute centroids
            np.mean(X[Ii], axis=0) for Ii in I
        ])
        Knext = K-Y.shape[0]  # How many points oversample in the future
        # Recursive oversampling
        if Knext > 0:
            return ReceptiveFieldFPS.naive_spherical_oversample(
                Knext, np.vstack([X, Y]), r, target_points, nthreads=nthreads
            )
        return np.vstack([X, Y])[:target_points]

    @staticmethod
    def gaussian_knn_oversample(K, X, k, nthreads=1):
        r"""
        Let :math:`\pmb{X} \in \mathbb{R}^{m \times n_x}` be a structure
        space matrix. If :math:`m < m^*`, then :math:`K = m^*-m` points
        must be generated through Gaussian k-nearest neighbors oversampling.
        This consists of finding the k-closest neighbors for each
        :math:`\pmb{x}_{i*}` and aggregating them using a Gaussian RBF to
        generate a new point.

        In case more new points than needed are generated, those with the
        smallest distance between :math:`\pmb{x}_{i*}` and its furthest
        nearest neighbor will be prioritized.

        The Gaussian RBF is calculated for each :math:`i`-th point as follows:

        .. math::

            \pmb{y}_{i*} = \dfrac{
                \sum_{\pmb{x}_{j*} \in \mathcal{N}_i} w_{ij} \pmb{x}_{j*}
            }{
                \sum_{\pmb{x}_{j*} \in \mathcal{N}_i} w_{ij}
            }

        Where :math:`\pmb{x}_{j*}` is the :math:`j`-th closest neighbor of
        :math:`\pmb{x}_{i*}`, :math:`\mathcal{N}_{i}` is the set of the
        k-nearest neighbors of :math:`\pmb{x}_{i*}`, and :math:`w_{ij}` is
        a Gaussian weight such that:

        .. math::

            w_{ij} = \exp\left[-\dfrac{
                \lVert{\pmb{x}_{j*} - \pmb{\mu}_{i*}}\rVert^{2}
            }{
                (d_i^*)^2
            }\right]

        With

        .. math::

            \pmb{\mu}_{i*} = k^{-1} \sum_{\pmb{x}_{j*} \in \mathcal{N}_i}{
                \pmb{x}_{j*}
            }

        and

        .. math::

            \left(d_i^*\right)^2 = \max \left\{{
                \lVert{\pmb{x}_{j*} - \pmb{\mu}_{i*}}\rVert^2 :
                1 \leq j \leq k
            }\right\}

        :param K: How many points must be sampled, i.e.,
            :math:`K = m^* - m`.
        :type K: int
        :param X: The structure space matrix.
        :type X: :class:`np.ndarray`
        :param k: The number of nearest neighbors to consider (:math:`k`)
        :type k: int
        """
        kdt = KDT(X)
        D, I = kdt.query(X, k=k, workers=nthreads)
        Knext = K-len(I)  # How many points oversample in the future
        Know = min(K, len(I))  # How many points oversample now
        S = np.argsort(D[:, -1])[:Know]  # Closest furthest pairs first
        XI = X[I[S]]  # The structure space for each knn neighborhood
        Y = np.mean(XI, axis=1)  # Compute centroids
        Dsq = np.sum(np.square(XI.transpose(1, 0, 2)-Y), axis=2)  # Sq. dist.
        dsq_max = np.max(Dsq, axis=0)  # Max squared distance
        W = np.exp(-Dsq/dsq_max)  # Weights
        Y = (  # Compute Gaussian centroids
            np.sum(W*XI.T, axis=1) / np.sum(W, axis=0)
        ).T
        # Recursive oversampling
        if Knext > 0:
            return ReceptiveFieldFPS.gaussian_knn_oversample(
                Knext, np.vstack([X, Y]), k, nthreads=nthreads
            )
        # Straightforward return
        return np.vstack([X, Y])

    def spherical_radiation_oversample(K, X, r, nthreads=1):
        r"""
        Let :math:`\pmb{X} \in \mathbb{R}^{m \times n_x}` be a structure space
        matrix. If :math:`m < m^*`, then :math:`K = m^* - m` points must be
        generated through spherical radiation oversampling. This consists of
        finding the spherical neighborhood centered on :math:`\pmb{x}_{i*}`,
        for each i-th point in the input structure space, and generating a new
        point by computing the Gaussian centroid of the neighborhood (see
        :meth:`.ReceptiveFieldFPS.gaussian_knn_oversample` for a detailed
        description on what Gaussian centroid means). The only difference
        for the Gaussian centroid with spherical radiation oversampling is
        that the radius of the sphere is used instead of the maximum squared
        distance such that:

        .. math::

            w_{ij} = \exp\left[-\dfrac{
                \lVert{\pmb{x}_{j*} - \pmb{\mu}_{i*}}\rVert^{2}
            }{
                r^2
            }\right]

        In case more new points than needed are generated, the furthest point
        sampling of the new points will be computed to select exactly
        :math:`m^*` points.

        :param K: How many points must be sampled, i.e.,
            :math:`K = m^* - m`.
        :type K: int
        :param X: The structure space matrix.
        :type X: :class:`np.ndarray`
        :param r: The radius for the spherical neighborhood :math:`r`.
        :type r: float
        """
        kdt = KDT(X)
        I = kdt.query_ball_tree(kdt, r)
        Y = [  # Compute centroids
            np.mean(X[Ii], axis=0) for Ii in I
        ]
        Dsq = [  # Squared distances
            np.sum(np.square(X[Ii]-Y[i]), axis=1)
            for i, Ii in enumerate(I)
        ]
        W = [np.exp(-Dsqi/(r*r)) for Dsqi in Dsq]  # Weights
        Y = np.array([  # Compute Gaussian centroids
            np.sum(W[i] * X[Ii].T, axis=1) / np.sum(W[i])
            for i, Ii in enumerate(I)
        ]).T
        Knext = K - Y.shape[0]  # How many points oversample in the future
        # Recursive oversampling
        if Knext > 0:
            return ReceptiveFieldFPS.spherical_radiation_oversample(
                Knext, np.vstack([X, Y]), r, nthreads=nthreads
            )
        # Reduce through FPS, if needed
        target_points = X.shape[0] + K
        if (X.shape[0] + Y.shape[0]) > target_points:
            o3d_cloud = open3d.geometry.PointCloud()
            o3d_cloud.points = open3d.utility.Vector3dVector(np.vstack([X, Y]))
            o3d_cloud = o3d_cloud.farthest_point_down_sample(target_points)
            if len(o3d_cloud.points) != target_points:
                raise ValueError(
                    'ReceptiveFieldFPS failed to reduce an spherical radiation '
                    'oversample. Probably due to repeated points.'
                )
            return np.asarray(o3d_cloud.points)
        # Straightforward return
        return np.vstack([X, Y])

    @staticmethod
    def naive_spherical_radiation_oversample(
        K, X, r, target_points, nthreads=1
    ):
        """
        Naive version of :meth:`.ReceptiveField.spherical_radiation_oversample`.
        See also :meth:`.ReceptiveField.naive_spherical_oversample` to
        understand the difference between naive and naive spherical
        approaches.
        """
        kdt = KDT(X)
        I = kdt.query_ball_tree(kdt, r)
        Y = [  # Compute centroids
            np.mean(X[Ii], axis=0) for Ii in I
        ]
        Dsq = [  # Squared distances
            np.sum(np.square(X[Ii]-Y[i]), axis=1)
            for i, Ii in enumerate(I)
        ]
        W = [np.exp(-Dsqi/(r*r)) for Dsqi in Dsq]  # Weights
        Y = np.array([  # Compute Gaussian centroids
            np.sum(W[i] * X[Ii].T, axis=1) / np.sum(W[i])
            for i, Ii in enumerate(I)
        ])
        Knext = K - Y.shape[0]  # How many points oversample in the future
        # Recursive oversampling
        if Knext > 0:
            return ReceptiveFieldFPS.naive_spherical_radiation_oversample(
                Knext, np.vstack([X, Y]), r, target_points, nthreads=nthreads
            )
        # Straightforward return
        return np.vstack([X, Y])[:target_points]

    # ---  MEMORY UTILS  --- #
    # ---------------------- #
    def canibalize(self, rf):
        """
        See :meth:`.ReceptiveField.canibalize`.
        """
        self.num_points = rf.num_points
        self.num_encoding_neighbors = rf.num_encoding_neighbors
        self.fast = rf.fast
        if rf.oversampling is not None:
            self.oversampling = dict(rf.oversampling)
        rf.oversampling = None
        self.N = rf.N
        rf.N = None
        self.M = rf.M
        rf.M = None
        self.x = rf.x
        rf.x = None
        self.Y = rf.Y
        rf.Y = None
        del rf
