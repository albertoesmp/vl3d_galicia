# ---   IMPORTS   --- #
# ------------------- #
from src.report.report import Report, ReportException
from src.inout.io_utils import IOUtils
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
from src.inout.point_cloud_io import PointCloudIO
from src.main.main_config import VL3DCFG
import src.main.main_logger as LOGGING
from scipy.spatial import KDTree as KDT
import numpy as np
import os


# ---   CLASS   --- #
# ----------------- #
class ReceptiveFieldOversamplingReport(Report):
    """
    :author: Alberto M. Esmoris Pena

    Class to handel reports related to receptive field oversampling.
    See :class:`.Report`.

    :ivar X: The structure space matrix of the oversampled receptive field.
    :vartype X: list of :class:`np.ndarray`
    :ivar Y: The structure space matrix for each receptive field, after the
        oversampling.
    :vartype Y: list of :class:`np.ndarray`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        r"""
        Initialize an instance of ReceptiveFieldOversamplingReport.

        :param kwargs: The key-word arguments.

        :Keyword Arguments:
            *   *X* (list ``np.ndarray``) --
                The structure space matrices before the oversampling.
            *   *Y* (list of ``np.ndarray``) --
                The structure space matrices after the oversampling.
            *   *id* (int or str or None) --
                See :meth:`.ReceptiveField.fit`.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign member attributes
        self.X = kwargs.get('X', None)
        if self.X is None:
            raise ReportException(
                'Receptive field oversampling report is not possible without '
                'the structure space matrices before the oversampling. To '
                'visualize the final receptive fields you can set '
                '"training_receptive_fields_dir" and/or '
                '"receptive_fields_dir" to generate a ReceptiveFieldsReport.'
            )
        self.Y = kwargs.get('Y', None)
        if self.Y is None:
            raise ReportException(
                'Receptive field oversampling report is not possible without '
                'the structure space matrix after the oversampling.'
            )
        self._id = kwargs.get('id', None)
        self.nthreads = kwargs.get('nthreads', -1)

    # ---   TO FILE   --- #
    # ------------------- #
    def to_file(self, path, out_prefix=None):
        """
        Write the report (oversampled receptive fields as point clouds) to
        files (LAZ).

        :param path: Path to the directory where the reports must be written.
        :type path: str
        :param out_prefix: The output prefix to expand the path (OPTIONAL).
        :type out_prefix: str
        :return: Nothing, the output is written to a file.
        """
        logging = False  # TODO Rethink : Make input argument to avoid multiprocessing LOGGING. See PointCloudFactoryFacade.make_from_arrays docs on logging argument
        # Expand path if necessary
        if out_prefix is not None and path[0] == "*":
            path = out_prefix[:-1] + path[1:]
        # Check given directory
        IOUtils.validate_path_to_directory(
            path,
            'Cannot find the directory to write the '
            'oversampled receptive fields :'
        )
        # Determine file path from dir
        if self._id is not None:
            path = os.path.join(path, f'OversampledRF_{self._id}.laz')
        else:
            path = os.path.join(path, 'OversampledRF.laz')
        # Mask oversampled points
        kdt = KDT(self.X)
        D = kdt.query(self.Y, 1, workers=self.nthreads)[0]
        mask = np.expand_dims((D != 0).astype(np.int8), 1)
        # Write output point cloud
        PointCloudIO.write(
            PointCloudFactoryFacade.make_from_arrays(
                self.Y,
                mask,
                fnames=['oversampled'],
                logging=logging
            ),
            path
        )
        # Log
        if logging:
            LOGGING.LOGGER.info(
                f'Oversampled receptive field written to "{path}"'
            )
