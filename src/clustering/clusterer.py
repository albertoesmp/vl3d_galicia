# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
import src.main.main_logger as LOGGING
from src.main.vl3d_exception import VL3DException
from src.utils.dict_utils import DictUtils
import time


# ---  EXCEPTIONS  --- #
# -------------------- #
class ClusteringException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to clustering components
    See :class:`.VL3DException`.
    """
    def __init__(self, message=''):
        # Call parent VL3DException
        super().__init__(message)


# ---   CLASS   --- #
# ----------------- #
class Clusterer:
    """
    :author: Alberto M. Esmoris Pena.

    Interface governing any clustering component.

    :ivar cluster_name: The name for the computed clusters. It will be used
        to reference the cluster column in the output point cloud.
    :vartype cluster_name: str
    :ivar post_clustering:
    :vartype post_clustering: None or list of callable
    """
    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_clustering_args(spec):
        """
        Extract the arguments to initialize/instantiate a Clusterer from a
        key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a Clusterer.
        """
        # Initialize
        kwargs = {
            'cluster_name': spec.get('cluster_name', None),
            'post_clustering': spec.get('post_clustering', None)
        }
        #  Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize a Clusterer.

        :param kwargs: The key-word arguments for the initialization of any
            Clusterer. It must contain the name of the cluster to be computed.
        """
        self.cluster_name = kwargs.get('cluster_name', 'CLUSTER')
        self.post_clustering = kwargs.get('post_clustering', None)

    # ---  CLUSTERING METHODS  --- #
    # ---------------------------- #
    def fit(self, pcloud):
        """
        Fit a clustering model to a given input point cloud.

        :param pcloud: The input point cloud to be used to fit the clustering
            model.
        :return: The clusterer itself, for fluent programming purposes.
        :rtype: :class:`.Clusteror`
        """
        return self

    @abstractmethod
    def cluster(self, pcloud):
        """
        Clustering from a given input point cloud.

        :param pcloud: The input point cloud for which clusters must be found.
        :return: The point cloud extended with the clusters.
        :rtype: :class:`.PointCloud`
        """
        pass

    def post_process(self, pcloud):
        """
        Run the post-processing pipeline on the given input point cloud.
        Clusters with

        :param pcloud: The input point cloud for the components in the
            post-processing pipeline.
        :return: The post-processed point cloud. Sometimes it will be exactly
            the same input point cloud because some post-processing components
            generate their output directly to a file.
        :rtype: :class:`.PointCloud`
        """
        # Ignore empty post-processing pipelines
        if self.post_clustering is None or len(self.post_clustering) < 1:
            return
        # Measure start time
        start = time.perf_counter()
        # Apply post-processing pipeline
        for callable in self.post_clustering:  # For each callable in the pipe.
            callable_start = time.perf_counter()
            pcloud = callable(self, pcloud)  # Run it
            callable_end = time.perf_counter()
            LOGGING.LOGGER.info(
                f'{self.__class__.__name__} computed '
                f'{callable.__class__.__name__} on {pcloud.get_num_points()} '
                f'points in {callable_end-callable_start:.3f} seconds.'
            )
        # Report time
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'{self.__class__.__name__} post-processed '
            f'{pcloud.get_num_points()} points '
            f'in {end-start:.3f} seconds.'
        )
        # Return
        return pcloud
