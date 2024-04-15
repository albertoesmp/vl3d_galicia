# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
from src.clustering.clusterer import ClusteringException


# ---   CLASS   --- #
# ----------------- #
class ClusteringPostProcessor:
    """
    :author: Alberto M. Esmoris Pena

    Interface governing any component of a clustering post-processing pipeline.
    See :class:`.Clusterer` and :meth:`.Clusterer.post_process`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize a ClusteringPostProcessor.

        :param kwargs: The key-word arguments for the initialization of any
            ClusteringPostProcessor.
        """
        pass

    # ---  POST-PROCESSING CALL  --- #
    # ------------------------------ #
    @abstractmethod
    def __call__(self, clusterer, pcloud):
        """
        Abstract method that must be overridden by any concrete (instantiable)
        component of a clustering post-processing pipeline.

        :param clusterer: The clusterer that called the post-processor.
        :param pcloud: The point cloud that must be post-processed.
        :return:
        """
        pass

    # ---   BUILD POST-PROCESSORS   --- #
    # --------------------------------- #
    @staticmethod
    def build_post_processor(spec):
        processor = spec.get('post-processor', None)
        processor_low = processor.lower()
        # TODO Rethink : Add post-processors here, e.g., bboxes from clusters
        raise ClusteringException(
            f'Unexpected post-processor: "{processor}"'
        )
