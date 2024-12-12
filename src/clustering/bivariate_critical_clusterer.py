# ---   IMPORTS   --- #
# ------------------- #
from src.clustering.clusterer import Clusterer, ClusteringException
import src.main.main_logger as LOGGING
from src.utils.dict_utils import DictUtils
import numpy as np
from scipy.spatial import KDTree as KDT
import shapely
import joblib
import tempfile
import os
import time


# ---   CLASS   --- #
# ----------------- #
class BivariateCriticalClusterer(Clusterer):
    r"""
    :author: Alberto M. Esmoris Pena

    Bivariate critical clustering consists of considering a third variable as
    a bivariate function of the first and second variables, e.g., the Monge
    form of a 3D surface :math:`z = \hat{z}(x, y)`. Then, it is possible
    to analyze the cylindrical neighborhood for each point in the point cloud
    :math:`\mathcal{N}(\pmb{x}_{i*}) = \left\{\pmb{x}_{k*} \in \mathcal{B}_{r}^{2}(\pmb{x}_{i*}) : 1 \leq k \leq m \right\}`,
    where :math:`\pmb{X} \in \mathbb{R}^{m \times 3}` would be a point cloud of
    :math:`m \in \mathbb{Z}_{>0}` points in three variables and
    :math:`\mathcal{B}_r^{2}(\pmb{x}_{i*})` is the ball of radius :math:`r`
    centered at the :math:`i`-th point of the point cloud in a 2D Euclidean
    space (considering only the first and second variables).

    First, a vector :math:`\pmb{z}^* \in \mathbb{R}^m` must be computed such
    that :math:`z^*_i = \max_{x_{k3}} \quad \mathcal{N}(\pmb{x}_{i*})`, i.e.,
    :math:`z^*_i` will be the max value of the third variable in the
    neighborhood of the :math:`i`-th point. Now, each point will be considered
    a maximum if and only if :math:`x_{i3} = z_{i}`. Alternatively, it
    can also be computed such that
    :math:`z^*_i = \min_{x_{k3}} \quad \mathcal{N}(\pmb{x}_{i*})`, that is
    why it is called critical clustering, because later on one cluster will be
    built for each critical point.

    There are two ways to cluster the points to its corresponding critical
    point. The first one is the **nearest neighbor method** and the second one
    is the **recursive** application of a Euclidean distance-based
    **region growing** algorithm.

    For the **nearest neighbor method**, each point is associated to the
    cluster of its closest neighbor in the space of the first two variables.

    For the **recursive region growing**, the cylindrical neighborhood
    of each critical point is considered for a given radius
    :math:`r_a \in \mathbb{R}_{>0}`. Then,
    for each new point added to the cluster its cylindrical neighborhood with
    radius :math:`r_b \in \mathbb{R}_{>0}` will be considered. Any point in
    this neighborhood will be added to the cluster and their neighborhoods will
    be recursively explored until no more points can be added.
    Optionally, a nearest neighbor
    correction can be applied to avoid having points without a cluster. It
    consists of applying the **nearest neighbor method** to any point that has
    not been clustered yet.

    :ivar precluster_name: The name of the attribute to be considered as the
        precluster. If None, then all points will be considered.
    :vartype cluster_name: str
    :ivar precluster_domain: The domain of the precluster, i.e., the precluster
        labels to be considered.
    :vartype precluster_domain: list
    :ivar critical_type: The type of critical point to be considered, either
        ``"min"`` or ``"max"``.
    :vartype critical_type: str
    :ivar radius: The radius :math:`r` for the neighborhoods when looking for
        critical points.
    :vartype radius: float
    :ivar filter_criticals: Whether to filter the critical points (``True``)
        or not (``False``). Critical points will be filtered such that in their
        neighborhoods there is only a single critical point, i.e., when there
        is more than one critical point in a neighborhood only one of them
        will be preserved. This scenario is will typically happen when there
        are at least two points at the same height in the neighborhood.
    :vartype filter_criticals: bool
    :ivar x: The name of the first variable.
    :vartype x: str
    :ivar y: The name of the second variable.
    :vartype y: str
    :ivar z: The name of the third variable.
    :vartype z: str
    :ivar strategy: The specification for the clustering strategy. It must be
        either a specification of the nearest neighbor method or the region
        growing algorithm.
    :vartype strategy: dict
    :ivar chunk_size: How many points per chunk for the parallel computations.
        If zero is given, then all the points will belong to the same chunk.
    :vartype chunk_size: int
    :ivar nthreads: How many chunks will be run in parallel at the same time.
        If :math:`-1` is given, then all threads will be used.
    :vartype nthreads: int
    :ivar kdt_nthreads: How many threads will be used to speedup the spatial
        queries through the KDTrees. Note that the final number of parallel
        jobs will be :math:`\text{nthreads} \times \text{kdt_nthreads}`.
    :vartype kdt_nthreads: int
    """
    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_clustering_args(spec):
        """
        Extract the arguments to initialize/instantiate a
        BivariateCriticalClusterer from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a
            BivariateCriticalClusterer.
        """
        # Extract arguments from parent
        kwargs = Clusterer.extract_clustering_args(spec)
        # Update arguments with those from BivariateCriticalClusterer
        kwargs['precluster_name'] = spec.get('precluster_name', None)
        kwargs['precluster_domain'] = spec.get('precluster_domain', None)
        kwargs['critical_type'] = spec.get('critical_type', None)
        kwargs['filter_criticals'] = spec.get('filter_criticals', None)
        kwargs['radius'] = spec.get('radius', None)
        kwargs['x'] = spec.get('x', None)
        kwargs['y'] = spec.get('y', None)
        kwargs['z'] = spec.get('z', None)
        kwargs['strategy'] = spec.get('strategy', None)
        kwargs['label_criticals'] = spec.get('label_criticals', None)
        kwargs['critical_label_name'] = spec.get('critical_label_name', None)
        kwargs['chunk_size'] = spec.get('chunk_size', None)
        kwargs['nthreads'] = spec.get('nthreads', None)
        kwargs['kdt_nthreads'] = spec.get('kdt_nthreads', None)
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return kwargs
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of BivariateCriticalClusterer.

        :param kwargs: The attributes of the BivariateCriticalClusterer that
            will also be passed to the parent.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign member attributes
        self.precluster_name = kwargs.get('precluster_name', None)
        self.precluster_domain = kwargs.get('precluster_domain', None)
        self.critical_type = kwargs.get('critical_type', 'max')
        self.filter_criticals = kwargs.get('filter_criticals', False)
        self.x = kwargs.get('x', 'x')
        self.y = kwargs.get('y', 'y')
        self.z = kwargs.get('z', 'z')
        self.radius = kwargs.get('radius', 0.3)
        self.strategy = kwargs.get('strategy', {
            "type": "NearestNeighbor"
        })
        self.label_criticals = kwargs.get('label_criticals', False)
        self.critical_label_name = kwargs.get(
            'critical_label_name', 'bcc_crits'
        )
        self.chunk_size = kwargs.get('chunk_size', 0)
        self.nthreads = kwargs.get('nthreads', 1)
        self.kdt_nthreads = kwargs.get('kdt_nthreads', 1)
        # Cache attributes
        self.dom_mask = None  # Cached domain mask
        # Validate member attributes
        if self.precluster_domain is not None:
            if not isinstance(self.precluster_domain, (list, tuple)):
                raise ClusteringException(
                    'BivariateCriticalClusterer does not support given '
                    f'precluster domain: {self.precluster_domain}'
                )

    # ---  CLUSTERING METHODS  --- #
    # ---------------------------- #
    def fit(self, pcloud):
        """
        The :class:`.BivariateCriticalClusterer` does not require any fit at
        all.
        See :class:`.Clusterer` and :meth:`.Clusterer.fit`.
        """
        return self

    def cluster(self, pcloud):
        """
        Apply bivariate critical clustering to the given point cloud.

        See :class:`.Clusterer` and :meth:`.Clusterer`
        """
        start = time.perf_counter()
        LOGGING.LOGGER.info('Computing bivariate critical clustering ...')
        # Get variables
        X = self.extract_variables(pcloud)
        pcloud.proxy_dump()
        m = X.shape[0]
        # Find critical points
        crit_mask = self.find_criticals(X)  # Boolean mask for critical points
        pcloud = self.add_critical_labels_to_point_cloud(pcloud, crit_mask)
        # Compute clusters
        c = self.compute_critical_clusters(X, crit_mask)
        # Consider points outside domain mask as not clustered
        X, crit_mask = None, None  # Not needed anymore
        if self.dom_mask is not None:
            _c = np.zeros_like(self.dom_mask, dtype=int)-1
            _c[self.dom_mask] = c
            c = _c
        # Report time
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'Bivariate critical clustering computed '
            f'{np.count_nonzero(np.unique(c) >= 0)} '
            f'clusters from {m} points in {end-start:.3f} seconds.'
        )
        # Return
        return self.add_cluster_labels_to_point_cloud(pcloud, c)

    # ---  BIVARIATE CRITICAL CLUSTERING METHODS  --- #
    # ----------------------------------------------- #
    def extract_variables(self, pcloud):
        """
        Extract relevant variables (those specified through self.x, self.y,
        and self.z) from the given point cloud.

        :param pcloud: The :class:`.PointCloud` from where the variables must
            be extracted.
        :type pcloud: :class:`.PointCloud`
        """
        # Function to extract a data from pcloud for a given variable
        def extract_variable(varname):
            if varname == 'x':
                return pcloud.get_coordinates_matrix()[:, 0]
            elif varname == 'y':
                return pcloud.get_coordinates_matrix()[:, 1]
            elif varname == 'z':
                return pcloud.get_coordinates_matrix()[:, 2]
            elif varname == 'classification':
                return pcloud.get_classes_vector()
            else:
                return pcloud.get_features_matrix(varname)
        # Handle default case (x,y,z) is the 3D structure space of the pcloud
        if self.x == 'x' and self.y == 'y' and self.z == 'z':
            X = pcloud.get_coordinates_matrix()
        else:
            # Handle general case
            X = np.array([
                extract_variable(self.x),
                extract_variable(self.y),
                extract_variable(self.z)
             ]).T
        # Filter out points outside specified domain
        self.dom_mask = None  # Release previous cached domain mask, if any
        if self.precluster_domain is not None:
            # TODO Rethink : Abstract pre-cluster logic also for DBScanClusterer
            # Get pre-clusters
            precluster_low = self.precluster_name.lower()
            if precluster_low == 'classification':
                y = pcloud.get_classes_vector()
            elif precluster_low in ['prediction', 'predictions']:
                y = pcloud.get_predictions_vector()
            else:
                y = pcloud.get_features_matrix(self.precluster_name)
            # Determine domain of preclusters
            y_dom = self.precluster_domain
            if y_dom is None:
                y_dom = np.unique(y)
            # Compute mask for points in the domain
            self.dom_mask = np.in1d(y, y_dom)
            X = X[self.dom_mask]
            LOGGING.LOGGER.info(
                f'BivariateCriticalClusterer found {X.shape[0]} points in '
                'the domain.'
            )
        # Return variables
        return X

    def find_criticals(self, X):
        """
        Find the critical points in the given point cloud, each representing
        a cluster.

        :param X: The 3D point cloud where the critical points must be found.
            The first and second column give the variables in the plane and
            the third column gives the variable for which critical points
            must be found.
        :type X: :class:`np.ndarray`
        """
        # Function to find the critical thresholds on a given chunk
        def find_critical_threshold_in_chunk(
            X2D_chunk, z, kdt, r, critical_f, kdt_nthreads
        ):
            I = kdt.query_ball_point(X2D_chunk, r, workers=kdt_nthreads)
            return [critical_f(z[Ii]) for Ii in I]
        # Start time measurement
        start = time.perf_counter()
        # Determine critical type
        crit_type_low = self.critical_type.lower()
        if crit_type_low == 'min':
            critical_f = np.min
        elif crit_type_low == 'max':
            critical_f = np.max
        else:
            raise ClusteringException(
                'BivariateCriticalClusterer does not recognize the critical '
                f'type: "{self.critical_type}"'
            )
        # Find threshold to determine critical points
        X2D = X[:, :2]  # The first and second variables
        z = X[:, 2]  # The third variable
        kdt = KDT(X2D)  # The KDTree
        m = X.shape[0]  # The number of points
        # Do the threshold finding in parallel
        if self.chunk_size > 0:
            chunk_size = self.chunk_size
            num_chunks = int(np.ceil(m/chunk_size))
        else:
            chunk_size = m
            num_chunks = 1
        zth = np.concatenate(joblib.Parallel(n_jobs=self.nthreads)(
            joblib.delayed(find_critical_threshold_in_chunk)(
                X2D[chunk_idx * chunk_size:min(m, (1+chunk_idx) * chunk_size)],
                z,
                kdt,
                self.radius,
                critical_f,
                self.kdt_nthreads
            ) for chunk_idx in range(num_chunks)
        ))
        # Find critical points
        crit_mask = z == zth
        # Report time
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'BivariateCriticalClusterer found {np.count_nonzero(crit_mask)} '
            f'critical points in {end-start:.3f} seconds.'
        )
        # Filter criticals, if requested (mostly for points at the same height)
        if self.filter_criticals:
            start = time.perf_counter()
            filter_count = 0
            crit_indices = np.flatnonzero(crit_mask)
            crit_X2D = X2D[crit_mask]
            kdt = KDT(crit_X2D)
            I = kdt.query_ball_point(  # Critical-only neighborhoods
                crit_X2D,
                self.radius,
                workers=self.kdt_nthreads*self.nthreads
            )
            for i, Ii in enumerate(I):
                if len(Ii) > 1:  # If more than one critical in neighborhood i
                    # Disable all the critical points but the first
                    crit_mask[crit_indices[Ii[1:]]] = False
                    filter_count += len(Ii)-1
                    for j in Ii[1:]:  # Dont explore disabled neighborhoods
                        I[j] = []
            end = time.perf_counter()
            LOGGING.LOGGER.info(
                f'BivariateCriticalClusterer filtered {filter_count} critical '
                f'points out in {end-start:.3f} seconds.'
            )
        # Return critical points
        return crit_mask

    def compute_critical_clusters(self, X, crit_mask):
        """
        Compute the final clusters according to the given strategy
        specification.

        See :meth:`BivariateCriticalClusterer.compute_nearest_neighbor_method`
        and :meth:`BivariateCriticalClusterer.compute_recursive_region_growing`
        .

        :param X: The 3D point cloud where from which points the clusters must
            be computed. The first and second column give the variables in the
            plane and the third column gives the variable for which critical
            points must be found.
        :type X: :class:`np.ndarray`
        """
        # Start time measurement
        start = time.perf_counter()
        # Determine strategy type
        strategy_type = self.strategy['type']
        type_low = strategy_type.lower()
        # Compute the critical clusters following the requested strategy
        if type_low == 'nearestneighbor':
            c = self.compute_nearest_neighbor_method(X, crit_mask)
        elif type_low == 'recursiveregiongrowing':
            c = self.compute_recursive_region_growing(X, crit_mask)
        else:
            raise ClusteringException(
                'BivariateCriticalClusterer failed to compute critical clusters '
                f'due to an unexepcted strategy type: "{strategy_type}"'
            )
        # Report time measurement
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'BivariateCriticalClusterer computed {np.count_nonzero(crit_mask)}'
            f' critical clusters using the {strategy_type} strategy in '
            f'{end-start:.3f} seconds.'
        )
        # Return clusters
        return c

    def add_critical_labels_to_point_cloud(self, pcloud, crit_mask):
        """
        Add labels to the critical points. Note that zero means the point is
        not a critical one.

        :param pcloud: The point cloud to which the critical labels must be
            added.
        :type pcloud: :class:`.PointCloud`
        :param crit_mask: The boolean mask where critical point are flagged as
            True.
        :type crit_mask: :class:`np.ndarray`
        :return: The input point cloud (updated inplace to add the critical
            labels).
        :rtype: :class:`.PointCloud`
        """
        if not self.label_criticals:  # Label critical points only if requested
            return pcloud
        # Label critical points with one, zero for non-critical
        if self.dom_mask is not None:  # Consider the change of domain, if any
            full_crit_mask = np.array(self.dom_mask)
            full_crit_mask[full_crit_mask] = \
                full_crit_mask[full_crit_mask] * crit_mask
            crits = np.zeros_like(full_crit_mask, dtype=np.int8)
            crits[full_crit_mask] = 1
        else:  # Case without change of domain
            crits = np.zeros_like(crit_mask, dtype=np.int8)
            crits[crit_mask] = 1
        # Update point cloud with critical labels
        pcloud.add_features(
            [self.critical_label_name],
            crits.reshape(-1, 1),
            ftypes=np.int8
        )
        pcloud.proxy_dump()
        # Return updated point cloud
        return pcloud

    @staticmethod
    def convert_cluster_labels_type(c, crit_mask=None):
        """
        Convert the integer type of the given cluster labels to use as few
        bytes as possible (on an element wise fashion) to represent the
        clusters.

        :param c: The point-wise cluster labels.
        :type c: :class:`np.ndarray`
        :param crit_mask: The point-wise boolean mask identifying the critical
            points. If given, the number of required bytes will be estimated
            from this mask instead of the cluster labels. If not given, the
            number of required bytes will be estimated from the cluster labels.
        :type crit_mask: :class:`np.ndarray` or None
        """
        required_bytes = np.max(c)+1 if crit_mask is None \
            else np.count_nonzero(crit_mask)+1
        if required_bytes < 128 and c.dtype != np.int8:
            return c.astype(np.int8)
        elif required_bytes < 32768 and c.dtype != np.int16:
            return c.astype(np.int16)
        elif required_bytes < 2147483648 and c.dtype != np.int32:
            return c.astype(np.int32)
        return c

    # ---  CLUSTERING STRATEGIES  --- #
    # ------------------------------- #
    def compute_nearest_neighbor_method(self, X, crit_mask):
        """
        Apply the nearest neighbor method to compute the clusters from the
        critical points.

        :param X: The point cloud to be clustered.
        :type X: :class:`np.ndarray`
        :param crit_mask: The boolean mask specifying whether a point is a
            critical point (true) or not (false).
        :type crit_mask: :class:`np.ndarray` of bool
        :return: The point-wise cluster labels.
        :rtype: :class:`np.ndarray`
        """
        # Do the nearest neighbor-based clustering
        X2D = X[:, :2]
        X2D_crit = X2D[crit_mask]
        kdt = KDT(X2D_crit)
        c = kdt.query(X2D, 1, workers=-1)[1]
        # Convert the int type of the labels to use as few bytes as possible
        return BivariateCriticalClusterer.convert_cluster_labels_type(c)

    def compute_recursive_region_growing(self, X, crit_mask):
        """
        Apply the recursive region growing algorithm to compute the clusters
        from the critical points.

        :param X: The point cloud to be clustered.
        :type X: :class:`np.ndarray`
        :param crit_mask: The boolean mask specifying whether a point is a
            critical point (true) or not (false).
        :type crit_mask: :class:`np.ndarray` of bool
        :return: The point-wise cluster labels.
        :rtype: :class:`np.ndarray`
        """
        # Extract strategy parameters
        max_iters = self.strategy.get('max_iters', 0)
        initial_radius = self.strategy.get('initial_radius', 0.1)  # r_a
        growing_radius = self.strategy.get('growing_radius', 0.2)  # r_b
        nn_correction = self.strategy.get('nn_correction', False)
        first_stage = self.strategy.get('first_stage', False)
        fs_correction = self.strategy.get('first_stage_correction', False)
        second_stage = self.strategy.get('second_stage', False)
        # Map max_iters = 0 to max int
        if max_iters == 0:
            max_iters = np.iinfo(int).max
        # Compute initial clusters
        X2D = X[:, :2]
        X2D_crit = X2D[crit_mask]
        kdt = KDT(X2D)
        I = kdt.query_ball_point(X2D_crit, initial_radius)
        c = np.zeros(X.shape[0], dtype=int) - 1
        c = BivariateCriticalClusterer.convert_cluster_labels_type(
            c, crit_mask=crit_mask
        )
        for i, Ii in enumerate(I):
            c[Ii] = i
        # Recursively expand clusters with first-stage clustering, if requested
        if first_stage:
            first_stage_start = time.perf_counter()
            c = self.compute_region_growing_first_stage(
                X2D, X[:, 2], c, max_iters, growing_radius, fs_correction
            )
            first_stage_end = time.perf_counter()
            LOGGING.LOGGER.debug(
                'BivariateCriticalClusterer computed first stage of region '
                f'growing in {first_stage_end-first_stage_start} seconds.'
            )
        # Recursively expand clusters through second-stage clustering
        if second_stage:
            second_stage_start = time.perf_counter()
            c = self.compute_region_growing_second_stage(
                c, X2D, growing_radius, max_iters
            )
            second_stage_end = time.perf_counter()
            LOGGING.LOGGER.debug(
                'BivariateCriticalClusterer computed second stage of region '
                f'growing in {second_stage_end-second_stage_start} seconds.'
            )
        # Apply nearest neighbor correction
        if nn_correction:
            nn_correction_start = time.perf_counter()
            clustered_mask = c >= 0
            X2D_clustered = X2D[clustered_mask]
            X2D_candidates = X2D[~clustered_mask]
            kdt = KDT(X2D_clustered)
            c[~clustered_mask] = c[clustered_mask][
                kdt.query(X2D_candidates, 1, workers=-1)[1]
            ]
            nn_correction_end = time.perf_counter()
            LOGGING.LOGGER.debug(
                'BivariateCriticalClusterer computed nearest neighbor '
                'correction after region growing in '
                f'{nn_correction_end-nn_correction_start} seconds.'
            )
        # Return clusters
        return c

    def compute_region_growing_first_stage(
        self, X2D, z, c, max_iters, growing_radius, correction
    ):
        """
        Compute the first stage of the region growing strategy. This first
        stage starts with finding the anti-critical points. If the searched
        critical points are the maximum, the anti-critical points will be the
        minimum, and vice versa. In the first stage, the clusters will grow
        by considering points in any neighborhood of a previously clustered
        point that is at the same height or below the center (if the critical
        points are maximum) or at the same height or above the center (if the
        critical points are minimum).

        Note that the output of the first stage can be corrected by computing
        the 2D concave hull of each cluster to consider any point inside this
        concave hull as part of the cluster.

        See
        :meth:`.BivariateCriticalClusterer.compute_recursive_region_growing`.

        :param X2D: The 2D structure space representing the point cloud (it
            includes the :math:`x` and :math:`y` coordinates).
        :type X2D: :class:`np.ndarray`
        :param z: The point-wise vertical coordinates.
        :type z: :class:`np.ndarray`
        :param c: The point-wise cluster labels.
        :type c: :class:`np.ndarray`
        :param max_iters: The maximum number of iterations for the first
            stage clustering.
        :type max_iters: int
        :param growing_radius: The radius for the first stage of the growing
            regions. The neighborhoods in the search space will be cylinders
            governed by this radius.
        :type growing_radius: float
        :param correction: Boolean flag governing whether the concave
            hull-based correction must be applied to the first stage clustering
            (``True``) or not (``False``).
        :type correction: bool
        """
        # Determine anti-critical type (e.g., min is anti critical of max)
        crit_type_low = self.critical_type.lower()
        if crit_type_low == 'min':
            anti_rel = lambda zk, zth: zk >= zth
        elif crit_type_low == 'max':
            anti_rel = lambda zk, zth: zk <= zth
        else:
            raise ClusteringException(
                'BivariateCriticalClusterer does not recognize the '
                'anti-critical type for the given critical type: '
                f'"{self.critical_type}"'
            )
        # Do the first-stage clustering itself
        non_ignore_mask = np.ones_like(c, dtype=bool)
        did_expand = True
        num_iters = 1
        while did_expand and num_iters < max_iters:
            did_expand = False
            num_iters += 1
            # Split clustered points from candidates (not clustered)
            clustered_mask = c >= 0
            non_ignore_clustered_mask = clustered_mask * non_ignore_mask
            candidate_indices = np.flatnonzero(~clustered_mask)
            X2D_clustered = X2D[non_ignore_clustered_mask]
            z_clustered = z[non_ignore_clustered_mask]
            X2D_candidates = X2D[~clustered_mask]
            z_candidates = z[~clustered_mask]
            kdt = KDT(X2D_candidates)  # KDT on candidate points
            # Find neighbors of clusters in candidate space
            I = kdt.query_ball_point(
                X2D_clustered, growing_radius,
                workers=self.nthreads * self.kdt_nthreads
            )
            c_clustered = c[non_ignore_clustered_mask]  # Non visited points
            for i, Ii in enumerate(I):
                if len(Ii) < 1:  # Skip empty neighborhoods
                    continue
                # Find points that satisfy the anti-critical condition
                accepted = np.flatnonzero(  # Indices of accepted candidates
                    anti_rel(z_candidates[Ii], z_clustered[i])
                )
                if len(accepted) > 0:  # Cluster them, if any
                    did_expand = True
                    c[candidate_indices[np.array(Ii)[accepted]]] = \
                        c_clustered[i]
            # Update non ignore mask
            non_ignore_mask[clustered_mask] = False
        # Apply first-stage correction based on concave hull
        if correction:
            # Prepare parallel cluster corrections
            cmax = np.max(c)
            non_clustered = np.flatnonzero(c < 0)
            non_clustered_X2D = X2D[non_clustered]
            kdt = KDT(non_clustered_X2D)  # KDT on non-clustered points
            nthreads = self.nthreads * self.kdt_nthreads  # Also num chunks
            chunk_size = int(np.ceil(cmax/(nthreads)))
            c_doms = [  # Clustering domain for each parallel task
                [i*chunk_size, min(cmax, (i+1)*chunk_size)]
                for i in range(nthreads)
            ]
            tmp_dir_path = tempfile.mkdtemp()
            tmp_file_path = os.path.join(tmp_dir_path, 'c.mmap')
            joblib.dump(c, tmp_file_path)
            c = joblib.load(tmp_file_path, mmap_mode='w+')
            # Compute many cluster corrections in parallel
            many_c = joblib.Parallel(n_jobs=nthreads)(
                joblib.delayed(
                    BivariateCriticalClusterer.apply_first_stage_correction
                )(
                    c,
                    c_doms[chunk_idx],
                    X2D,
                    non_clustered,
                    non_clustered_X2D,
                    kdt
                ) for chunk_idx in range(nthreads)
            )
            # Overwrite previous clustering
            for ci in many_c:  # On overlapping, the last case will prevail
                mask = ci >= 0
                c[mask] = ci[mask]
        # Return clusters from first stage
        return c

    @staticmethod
    def apply_first_stage_correction(
        c, c_dom, X2D, non_clustered, non_clustered_X2D, kdt
    ):
        """
        Method that chunk-wise applies the concave hull-based correction to the
        first stage of the region growing clustering.

        See
        :meth:`.BivariateCriticalClusterer.compute_region_growing_first_stage`.

        :param c: The point-wise cluster labels.
        :type c: :class:`np.ndarray`
        :param c_dom: The :math:`[a, b]` interval of cluster labels defining
            the domain for the current chunk.
        :type c_dom: list
        :param X2D: The 2D structure space representing the point cloud (it
            includes the :math:`x` and :math:`y` coordinates).
        :type X2D: :class:`np.ndarray`
        :param non_clustered: The array with the indices of points that do not
            belong to a cluster yet.
        :type non_clustered: :class:`np.ndarray`
        :param non_clustered_X2D: The 2D structure space representing the
            non-clustered points.
        :type non_clustered_X2D: :class:`np.ndarray`
        :param kdt: The KDTree built on the non-clustered 2D structure space.
        """
        for ci in range(c_dom[0], c_dom[1]+1):
            # Find concave hull
            ci_X2D = X2D[c == ci]  # Select points in cluster
            if len(ci_X2D) < 1:  # Skip empty clusters
                continue
            hull = shapely.concave_hull(shapely.MultiPoint(ci_X2D))
            # Get all non-clustered points in the cluster neighborhood
            ci_X2D_min = np.min(ci_X2D, axis=0)
            ci_X2D_max = np.max(ci_X2D, axis=0)
            cent = (ci_X2D_min+ci_X2D_max)/2.0  # Center point (midrange)
            rang = np.max(ci_X2D_max-ci_X2D_min)  # Max coordinate range
            I = kdt.query_ball_point(cent, rang/2, workers=1)
            # Cluster non-clustered points inside the concave hull
            for i in I:
                if hull.contains(shapely.Point(non_clustered_X2D[i])):
                    c[non_clustered[i]] = ci
        # Return corrected clusters
        return c

    def compute_region_growing_second_stage(
        self, c, X2D, growing_radius, max_iters
    ):
        """
        Compute the second stage of the region growing strategy. This second
        stage consists of growing the clusters by considering the neighborhood
        of each point in the cluster and adding the points in this neighborhood
        to the cluster until it is not possible to grow further considering
        the radius governing the neighborhood.

        See
        :meth:`.BivariateCriticalClusterer.compute_recursive_region_growing`.

        :param c: The point-wise cluster labels.
        :type c: :class:`np.ndarray`
        :param X2D: The 2D structure space representing the point cloud (it
            includes the :math:`x` and :math:`y` coordinates).
        :type X2D: :class:`np.ndarray`
        :param growing_radius: The radius for the first stage of the growing
            regions. The neighborhoods in the search space will be cylinders
            governed by this radius.
        :type growing_radius: float
        :param max_iters: The maximum number of iterations for the first
            stage clustering.
        :type max_iters: int
        """
        non_ignore_mask = np.ones_like(c, dtype=bool)
        did_expand = True
        num_iters = 1
        while did_expand and num_iters < max_iters:
            did_expand = False
            num_iters += 1
            # Split clustered points from candidates (not clustered)
            clustered_mask = c >= 0
            non_ignore_clustered_mask = clustered_mask * non_ignore_mask
            candidate_indices = np.flatnonzero(~clustered_mask)
            X2D_clustered = X2D[non_ignore_clustered_mask]
            X2D_candidates = X2D[~clustered_mask]
            kdt = KDT(X2D_candidates)  # KDT on candidate points
            # Find neighbors of clusters in candidate space
            I = kdt.query_ball_point(
                X2D_clustered, growing_radius,
                workers=self.nthreads * self.kdt_nthreads
            )
            c_clustered = c[non_ignore_clustered_mask]  # Non visited points
            for i, Ii in enumerate(I):
                if len(Ii) > 0:
                    did_expand = True
                    c[candidate_indices[Ii]] = c_clustered[i]
                else:
                    continue
            # Update non ignore mask
            non_ignore_mask[clustered_mask] = False
        # Return clusters from second stage
        return c
