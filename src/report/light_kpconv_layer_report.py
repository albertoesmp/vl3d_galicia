# ---   IMPORTS   --- #
# ------------------- #
from src.report.report import Report, ReportException
from src.inout.io_utils import IOUtils
import numpy as np
import os


# ---   CLASS   --- #
# ----------------- #
class LightKPConvLayerReport(Report):
    """
    :author: Alberto M. Esmoris Pena

    Class to handle reports that represent a light KPConv layer.

    See :class:`.Report` and :class:`.KPConvLayerReport`.
    See also :class:`.LightKPConvLayer`.

    :ivar Q: The matrix of the kernel's structure space.
    :vartype Q: :class:`np.ndarray`
    :ivar W: The matrix representing the weights.
    :vartype W: :class:`np.ndarray`
    :ivar A: The matrix of scale factors.
    :vartype A: :class:`np.ndarray`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, Q, W, A, **kwargs):
        """
        Initialize an instance of LightKPConvLayerReport.

        :param Q: The kernel's structure space.
        :type Q: :class:`np.ndarray`
        :param W: The kernel's weights.
        :type W: :class:`np.ndarray`
        :param A: The scale factors of the layer.
        :type A: :class:`np.ndarray`
        :param kwargs: The key-word arguments.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.Q = Q
        self.W = W
        self.A = A
        # Validate
        if self.Q is None or len(self.Q) < 1:
            raise ReportException(
                'LightKPConvLayerReport did not receive the structure of the '
                'kernel.'
            )
        if self.W is None or len(self.W) < 1:
            raise ReportException(
                'LightKPConvLayerReport did not receive the weights of the '
                'kernel.'
            )
        if self.A is None or len(self.A) < 1:
            raise ReportException(
                'LightKPConvLayerReport did not receive the scale factors of '
                'the kernel.'
            )

    # ---   TO FILE   --- #
    # ------------------- #
    def to_file(self, path, out_prefix=None):
        """
        Write the report (structure and weights) to files.

        :param path: Path to the directory where the report files must be
            written.
        :type path: str
        :param out_prefix: The output prefix to expand the path (OPTIONAL).
        :type out_prefix: str
        :return: Nothing, the output is written to a file.
        """
        # Expand path if necessary
        if out_prefix is not None and path[0] == '*':
            path = out_prefix[:-1] + path[1:]
        # Check
        IOUtils.validate_path_to_directory(
            path,
            'Cannot find the directory to write the light KPConv '
            'layer representation.'
        )
        # Output kernel's structure
        np.savetxt(
            os.path.join(path, 'Q.csv'),
            self.Q,
            fmt='%.7f'
        )
        # Output kernel's weights
        np.savetxt(
            os.path.join(path, f'W.csv'),
            self.W,
            fmt='%.7f'
        )
        # Output kernel's scale factors
        np.savetxt(
            os.path.join(path, f'A.csv'),
            self.A,
            fmt='%.7f'
        )
