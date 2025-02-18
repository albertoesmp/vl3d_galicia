# ---   IMPORTS   --- #
# ------------------- #
from src.clustering.postproc.clustering_post_processor import \
    ClusteringPostProcessor, ClusteringException
import src.main.main_logger as LOGGING
from scipy.spatial import KDTree as KDT
import numpy as np
import time

# TODO Rethink : Implement

class ClusterCorrector(ClusteringPostProcessor):
    """
    :author: Alberto M. Esmoris Pena

    Clustering post-processor that computes corrections on the clusters.
    See :class:`.ClusteringPostProcessor`.

    :ivar correction_conditions: The many conditions that must be satisfied
        (all of them, i.e., and of the list's elements) for any cluster that
        needs the correction.
    :vartype correction_conditions: list of dict
    :ivar correction_action: The action that must be performed on any cluster
        that needs correction.
    :vartype correction_Action: str
    :ivar merge_conditions: The many conditions that must be satisfied when
        the correction action is merge-based. If these conditions are not
        satisfied, the alternative handling of the cluster will be computed
        instead of the merge.
    :vartype merge_conditions: list of dict
    :ivar break_conditions: The many conditions that must be satisfied when
        the correction action is break-based. If these conditions are not
        satisfied, the alternative handling of the cluster will be computed
        instead of the merge.
    :vartype break_conditions: list of dict
    :ivar cache: Key-value pairs where the key identifies a relevant value
        that can be used at more than one point during the execution of the
        cluster corrector.
    :vartype cache: dict
    :ivar nthreads: The number of threads to be used for parallel computations.
        Note that -1 means using as many threads as available cores.
    :vartype nthreads: int
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize a ClusterCorrector post-processor.

        See :meth:`.ClusteringPostProcessor.__init__`.

        :param kwargs: The key-word arguments for the initialization of the
            ClusterCorrector.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign member attributes
        self.correction_conditions = kwargs.get('correction_conditions', None)
        self.correction_action = kwargs['correction_action']
        self.merge_conditions = kwargs.get('merge_conditions', None)
        self.break_conditions = kwargs.get('break_conditions', None)
        self.nthreads = kwargs.get('nthreads', 1)

    # ---  POST-PROCESSING CALL  --- #
    # ------------------------------ #
    def __call__(self, clusterer, pcloud, out_prefix=None):
        """
        Post-process the given point cloud with clusters to compute the
        corresponding cluster corrections.

        :param clusterer: The clusterer that generated the clusters.
        :type clusterer: :class:`.Clusterer`
        :param pcloud: The point cloud to be post-processed.
        :type pcloud: :class:`.PointCloud`
        :param out_prefix: The output prefix in case path expansion must be
            applied.
        :type out_prefix: str or None
        :return: The post-processed point cloud.
        :rtype: :class:`.PointCloud`
        """
        # Prepare post-processing
        start = time.perf_counter()
        X = pcloud.get_coordinates_matrix()
        c = self.get_cluster_labels(clusterer, pcloud)
        c_dom = np.unique(c[c > -1])  # All unique values but non clusters (-1)
        cache = {}  # The cache for memoization-based optimizations
        # Find correction domain (i.e., the clusters that need to be corrected)
        # TODO Rethink : Implement
        start_cordom = time.perf_counter()
        c_cordom = self.find_correction_domain(X, c, c_dom, cache)
        end_cordom = time.perf_counter()
        LOGGING.LOGGER.info(
            f'ClusterCorrector found a correction domain of {len(c_cordom)} '
            f'clusters in {end_cordom-start_cordom:.3f} seconds.'
        )
        # Compute min set distance between each cluster and its closest cluster
        start_fcp = time.perf_counter()
        ClusterCorrector.find_closest_pairs(
            X, c, c_dom, c_cordom, cache, nthreads=self.nthreads
        )
        end_fcp = time.perf_counter()
        LOGGING.LOGGER.info(
            'ClusterCorrector found closest pairs in '
            f'{end_fcp - start_fcp:.3f} seconds.'
        )
        # Precompute values for cluster correction
        start_precomp = time.perf_counter()
        # TODO Rethink : Implement
        end_precomp = time.perf_counter()
        LOGGING.LOGGER.info(
            'ClusterCorrector did precomputations in '
            f'{end_precomp-start_precomp:.3f} seconds.'
        )
        # Compute correction for each cluster
        start_corrections = time.perf_counter()
        num_corrections = 0
        # TODO Rethink : Implement
        end_corrections = time.perf_counter()
        LOGGING.LOGGER.info(
            'ClusterCorrector computed corrections from precomputed values '
            f'in {end_corrections-start_corrections:.3f} seconds.'
        )
        # Report time
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'ClusterCorrector computed {num_corrections} corrections on '
            f'{len(c_dom)} clusters in {end-start:.3f} seconds.'
        )
        # Return
        return pcloud

    # ---  CLUSTER CORRECTOR METHODS  --- #
    # ----------------------------------- #
    @staticmethod
    def find_closest_pairs(X, c, c_dom, c_cordom, cache, nthreads=1):
        """
        For each cluster find its closest cluster. The results will be inserted
        into the cache in two different elements. The first one will be a
        dictionary identified by the key "closest_cluster_label". The keys will
        be the labels of the different clusters and the value the label of the
        closest cluster. The second one will be a dictionary identified by the
        key "closest_cluster_distance" and the value will be the minimum
        distance between the set of points representing the first cluster and
        the set of points representing the second cluster.

        :param X: The structure space matrix representing the point cloud.
        :type X: :class:`np.ndarray`
        :param c: The point-wise cluster labels.
        :type c: :class:`np.ndarray`
        :param c_dom: The different cluster labels.
        :type c_dom: :class:`np.ndarray`
        :param c_cordom: The labels of the clusters that need to be corrected.
        :type c_cordom: :class:`np.ndarray`
        :param cache: The cache where the results will be stored.
        :type cache: dict
        :param nthreads: The number of threads for the parallel computations.
            If -1 is given, as many threads as available cores will be used.
        :type nthreads: int
        :return: The updated cache itself (it is also updated inplace).
        :rtype: dict
        """
        # TODO Rethink : Elliptical acceleration, i.e., abstract clusters to ellipses and find min distance between ellipses instead of point-wise
        cwX = {}  # Dictionary of cluster-wise structure spaces
        cache["closest_cluster_label"] = {}
        cache["closest_cluster_distance"] = {}
        tmp = {}  # Temporary cache for distances
        for cl in c_dom:  # For each cluster
            cwX[cl] = X[c == cl]  # Get points in cluster
        for ck in c_cordom:  # For each k-th cluster that needs a correction
            kKdt = KDT(cwX[ck])
            dmin = np.finfo(X.dtype).max
            cmin = -1
            for cl in c_dom:  # For each l-th cluster (considering all)
                if ck == cl:  # Ignoring k=l, i.e., avoid cluster reflections
                    continue
                # Check if ck, cl has been temporarily cached
                if ck in tmp and cl in tmp[ck]:
                    dkl = tmp[ck][cl]
                # Also check if cl, ck has been temporarily cached
                elif cl in tmp and ck in tmp[cl]:
                    dkl = tmp[cl][ck]
                # Otherwise, do the computation and insert it into tmp. cache
                else:
                    dkl = ClusterCorrector.calc_min_set_distance_kdt(
                        cwX[cl], kKdt, nthreads=nthreads
                    )
                    if ck not in tmp:
                        tmp[ck] = {}
                    tmp[ck][cl] = dkl
                if dkl < dmin:  # Take the min set distance cluster
                    dmin = dkl
                    cmin = cl
            # Store results in memoization cache
            cache["closest_cluster_label"][ck] = cmin
            cache["closest_cluster_distance"][ck] = dmin
        return cache  # Return updated memoization cache (note in-place update)

    def find_correction_domain(self, X, c, c_dom, cache):
        """
        Find the correction domain, i.e, the labels of the clusters that need
        to be corrected. In doing so, the cache is used to store relevant
        values to speedup the computations through memoization.

        :param X: The structure space representing the point cloud.
        :type X: :class:`np.ndarray`
        :param c: The point-wise cluster labels.
        :type c: :class:`np.ndarray`
        :param c_dom: The domain of cluster labels.
        :type c_dom: :class:`np.ndarray`
        :param cache: A cache used for memoization-based performance
            optimizations.
        :type cache: dict
        :return: The correction domain that must be subset or equal to the
            cluster domain.
        :rtype: :class:`np.ndarray`
        """
        c_cordom = []  # Initial correction domain
        for ck in c_dom:  # For each k-th cluster (with label ck)
            Xk = X[c == ck]  # Cluster's structure space
            needs_correction = True
            for cond in self.correction_conditions:  # For each cor. cond.
                needs_correction = (
                    needs_correction and
                    ClusterCorrector.check_correction_condition(  # TODO Rethink : Implement
                        cond,
                        X,
                        Xk,
                        c,
                        ck,
                        cache
                    )
                )
            if needs_correction:  # Add to correction domain if needed
                c_cordom.append(ck)
        # Return cluster correction domain as array
        return np.array(c_cordom, dtype=c.dtype)

    # ---  CORRECTION CONDITIONS METHODS  --- #
    # --------------------------------------- #

    # ---  MERGE CONDITIONS METHODS  --- #
    # ---------------------------------- #

    # ---  BREAK CONDITIONS METHODS  --- #
    # ---------------------------------- #

    # ---  CORRECTION ACTION METHODS  --- #
    # ----------------------------------- #

    # ---   CALC METHODS   --- #
    # ------------------------ #
    @staticmethod
    def calc_min_set_distance(Xk, Xl):
        r"""
        Compute the minimum distance between the two given structure spaces.

        .. math::
            \min_{ij} \; \lVert(\pmb{X_k})_{j*} - (\pmb{X_k})_{i*}\rVert
        """
        return np.sqrt(np.min(  # Return min Euclidean distance
            np.sum(np.square(  # Squared Euclidean distances
                np.expand_dims(Xk, axis=1) - np.expand_dims(Xl, axis=0),
            ), axis=2)
        ))

    @staticmethod
    def calc_min_set_distance_kdt(Xl, kKdt, nthreads=1):
        """
        Like :meth:`.ClusterCorrector.calc_min_set_distance` but using a KDT
        built on the :math:`k`-th structure space to speedup the min distance
        computation.
        """
        return np.min(kKdt.query(Xl, 1, workers=nthreads)[0])


