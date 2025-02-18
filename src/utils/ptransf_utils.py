# ---   IMPORTS   --- #
# ------------------- #
from src.utils.ptransf.point_cloud_sampler import PointCloudSampler


# ---   CLASS   --- #
# ----------------- #
class PtransfUtils:
    """
    :author: Alberto M. Esmoris Pena

    Class with util static methods to work with point transformers.
    """

    # ---  EXTRACT FROM SPEC  --- #
    # --------------------------- #
    @staticmethod
    def extract_ptransf_class(spec):
        """
        Extract the point transformer's class from the key-word
        specification.

        :param spec: The key-word specification.
        :return: Class representing/realizing a point transformer.
        :rtype: :class:`.PointTransformer`
        """
        ptransf = spec.get('point_transformer', None)
        if ptransf is None:
            raise ValueError(
                'Transforming points requires a point transformer. None '
                'was specified.'
            )
        # Check point transformer class
        ptransf_low = ptransf.lower()
        if ptransf_low == 'pointcloudsampler':
            return PointCloudSampler
        # An unknown point transformer was specified
        raise ValueError(f'There is no known point transformer: "{ptransf}"')
