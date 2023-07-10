# ---   IMPORTS   --- #
# ------------------- #
from src.pipeline.pipeline import Pipeline, PipelineException
import src.main.main_logger as LOGGING
from src.main.main_mine import MainMine
from src.main.main_train import MainTrain
from src.mining.miner import Miner
from src.utils.imput.imputer import Imputer
from src.utils.imputer_utils import ImputerUtils
from src.utils.ftransf.feature_transformer import FeatureTransformer
from src.utils.ftransf_utils import FtransfUtils
from src.model.model_op import ModelOp
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
from src.inout.writer import Writer
from src.inout.model_writer import ModelWriter
from src.inout.writer_utils import WriterUtils
import time


# ---   CLASS   --- #
# ----------------- #
class SequentialPipeline(Pipeline):
    """
    :author: Alberto M. Esmoris Pena

    Sequential pipeline (no loops, no recursion).
    See :class:`Pipeline`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of SequentialPipeline.
        A sequential pipeline execute the different components in the order
        they are given.
        See parent :class:`SequentialPipeline`

        :param kwargs: The attributes for the SequentialPipeline
        :ivar sequence: The sequence of components defining the
            SequentialPipeline.
        :vartype sequence: list
        """
        # Call parent init
        super().__init__(**kwargs)
        # Pipeline as sequential list of components
        self.sequence = []
        seq_pipeline = kwargs.get('sequential_pipeline', None)
        if seq_pipeline is None:
            raise PipelineException(
                'Critical error initializing SequentialPipeline.\n'
                'There is no sequential_pipeline specification.\n'
                'This MUST not happen. Please, contact the developers.'
            )
        for comp in seq_pipeline:
            # Handle each potential component type
            if comp.get("miner", None) is not None:  # Handle miner
                miner_class = MainMine.extract_miner_class(comp)
                miner = miner_class(**miner_class.extract_miner_args(comp))
                self.sequence.append(miner)
            if comp.get("imputer", None) is not None:  # Handle imputer
                imputer_class = ImputerUtils.extract_imputer_class(comp)
                imputer = imputer_class(
                    **imputer_class.extract_imputer_args(comp)
                )
                self.sequence.append(imputer)
            if comp.get("feature_transformer", None) is not None:  # Hdl. ftr.
                ftransf_class = FtransfUtils.extract_ftransf_class(comp)
                ftransf = ftransf_class(
                    **ftransf_class.extract_ftransf_args(comp)
                )
                self.sequence.append(ftransf)
            if comp.get("train", None) is not None:  # Handle train
                model_class = MainTrain.extract_model_class(comp)
                model = model_class(**model_class.extract_model_args(comp))
                self.sequence.append(ModelOp(model, ModelOp.OP.TRAIN))
            # TODO Rethink : Add elif for train, predict, and eval too
            # Handle writer as component
            if comp.get("writer", None) is not None:  # Handle writer
                writer_class = WriterUtils.extract_writer_class(comp)
                writer = writer_class(comp.get("out_pcloud"))
                self.sequence.append(writer)
            # Handle potential writer for current non-writer component
            elif comp.get('out_pcloud', None) is not None:
                self.sequence.append(Writer(comp['out_pcloud']))

    # ---  RUN PIPELINE  --- #
    # ---------------------- #
    def run(self):
        """
        Run the sequential pipeline.

        :return: Nothing.
        """
        # List of input point clouds (even if just one is given)
        in_pclouds = self.in_pcloud
        if isinstance(in_pclouds, str):
            in_pclouds = [in_pclouds]
        # For each input point cloud, run the pipeline
        for i, in_pcloud in enumerate(in_pclouds):
            # Handle path to output point cloud, if any
            out_pcloud = None
            if self.out_pcloud is not None:
                out_pcloud = self.out_pcloud[i]
            # Run the pipeline for the current case
            self.run_case(in_pcloud, out_pcloud=out_pcloud)

    def run_case(self, in_pcloud, out_pcloud=None):
        """
        Run the sequential pipeline for a particular input point cloud.

        :param in_pcloud: The input point cloud for this particular case.
        :param out_pcloud: Optionally, the output path or prefix.
        :return: Nothing.
        """
        LOGGING.LOGGER.info(
            f'SequentialPipeline running for "{in_pcloud}" ...'
        )
        start = time.perf_counter()
        # Load input
        pcloud = PointCloudFactoryFacade.make_from_file(in_pcloud)
        model = None  # TODO Rethink : Handle model loading (if any)
        # Run pipeline
        for comp in self.sequence:
            if isinstance(comp, Miner):  # Handle miner
                pcloud = comp.mine(pcloud)
            elif isinstance(comp, Imputer):  # Handle imputer
                pcloud = comp.impute_pcloud(pcloud)
            elif isinstance(comp, FeatureTransformer):  # Handle feat. transf.
                pcloud = comp.transform_pcloud(pcloud, out_prefix=out_pcloud)
            elif isinstance(comp, ModelOp) and comp.op == ModelOp.OP.TRAIN:
                # Handle train
                model = comp(pcloud, out_prefix=out_pcloud)
            # TODO Rethink : Add elif for train, predict, and eval too
            elif isinstance(comp, Writer):  # Handle writer
                if comp.needs_prefix():
                    if out_pcloud is None:
                        raise PipelineException(
                            "SequentialPipeline is running a case with no "
                            "path prefix for the output.\n"
                            "This requirement is a MUST."
                        )
                    if isinstance(comp, ModelWriter):
                        comp.write(model, prefix=out_pcloud)
                    else:
                        comp.write(pcloud, prefix=out_pcloud)
                else:
                    comp.write(pcloud)
        # If the pipeline's out_pcloud is a full path, use it automatically
        if out_pcloud is not None and out_pcloud[-1] != "*":
            writer = Writer(out_pcloud)
            if not writer.needs_prefix():
                writer.write(pcloud)
        # Report execution time
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'SequentialPipeline for "{in_pcloud}" '
            f'computed in {end-start:.3f} seconds.'
        )

