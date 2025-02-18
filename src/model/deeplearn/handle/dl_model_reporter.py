# ---   IMPORTS   --- #
# ------------------- #
from src.report.deep_learning_model_summary_report import \
    DeepLearningModelSummaryReport
from src.report.training_history_report import TrainingHistoryReport
from src.plot.training_history_plot import TrainingHistoryPlot
from src.report.receptive_fields_report import ReceptiveFieldsReport
from src.report.receptive_fields_distribution_report import \
    ReceptiveFieldsDistributionReport
from src.plot.receptive_fields_distribution_plot import \
    ReceptiveFieldsDistributionPlot
from src.model.deeplearn.arch.spconv3d_pwise_classif import SpConv3DPwiseClassif
import src.main.main_logger as LOGGING
import tensorflow as tf
from tensorflow.python.framework.errors_impl import ResourceExhaustedError as \
    TFResourceExhaustedError, InternalError as TFInternalError
import numpy as np
import os


# ---   CLASS   --- #
# ----------------- #
class DLModelReporter:
    """
    Class to handle plots and reports for deep learning models.
    Note that the plots and reports that are responsibility of this handler
    are those involved in the operations directly handled by a
    :class:`.DLModelHandler` or any derived class like the
    :class:`.SimpleDLModelHandler`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        pass

    # ---  MODEL SUMMARY REPORT  --- #
    # ------------------------------ #
    def handle_model_summary_report(self, mh):
        """
        Write the model's summary and also print it through logging system.

        This method requires that the model has been compiled so the summary
        is available.

        :param mh: The model handler whose model's summary must be reported.
        :return: Nothing at all.
        """
        summary = DeepLearningModelSummaryReport(mh.compiled)
        LOGGING.LOGGER.info(summary.to_string())
        if mh.path_manager.summary_report_path is not None:
            summary.to_file(
                mh.path_manager.summary_report_path,
                out_prefix=mh.out_prefix
            )
            LOGGING.LOGGER.info(
                'Deep learning model summary written to "'
                f'{mh.path_manager.summary_report_path}"'
            )

    # ---  HISTORY PLOTS AND REPORTS  --- #
    # ----------------------------------- #
    def handle_history_plots_and_reports(self, mh):
        # TODO Rethink : Doc
        # Check a directory to write the training history data has been given
        if mh.path_manager.training_history_dir is None:
            return
        if mh.history is None or len(mh.history.history) < 1:
            return
        # Do the plots and reports
        report_path = os.path.join(
            mh.path_manager.training_history_dir, 'training_history.csv'
        )
        TrainingHistoryReport(
            mh.history
        ).to_file(report_path, out_prefix=mh.out_prefix)
        LOGGING.LOGGER.info(
            'Deep learning training history report written to '
            f'"{report_path}"'
        )
        TrainingHistoryPlot(
            mh.history,
            path=mh.path_manager.training_history_dir
        ).plot(
            out_prefix=mh.out_prefix
        )
        LOGGING.LOGGER.info(
            'Deep learning training history plots exported to '
            f'"{mh.path_manager.training_history_dir}"'
        )


    # ---  RECEPTIVE FIELD PLOTS AND REPORTS  --- #
    # ------------------------------------------- #
    def handle_receptive_fields_plots_and_reports(
        self, mh, X_rf, X=None, y=None
    ):
        """
        Handle the plot and reports related to receptive fields including the
        decision on whether they must be plotted. This method also computes
        the necessary data (e.g., the model's probabilities). Note that this
        method must only be called during fit. To plot and report the
        data of the receptive fields call the
        :meth:`.DLModelReporter.do_receptive_fields_plots_and_reports`
        method directly.

        :param mh: The model handler whose receptive fields' plots and reports
            must be done.
        :type mh: :class:`.DLModelHandler`
        :param X_rf: The input for the model as computed by the model's
            pre-processor.
        :param X: The structure space representing the original point cloud
            (not the receptive fields).
        :param y: The vector of expected labels, the ground-truth from the
            supervised training perspective.
        :type y: :class:`np.ndarray`
        :return: Nothing at all.
        """
        # Check the model handler has a pre-processor
        if getattr(mh.arch.pre_runnable, 'pre_processor', None) is None:
            return
        # Check at least one training receptive field report (or plot) has
        # been requested
        if getattr(
            mh.arch.pre_runnable.pre_processor,
            'training_receptive_fields_dir',
            None
        ) is None and getattr(
            mh.arch.pre_runnable.pre_processor,
            'training_receptive_fields_distribution_report_path',
            None
        ) is None and getattr(
            mh.arch.pre_runnable.pre_processor,
            'training_receptive_fields_distribution_plot_path',
            None
        ) is None:
            return
        # Prepare the receptive fields' plots and reports
        try:
            zhat = mh.predict_rf(X_rf)
        except (TFResourceExhaustedError, TFInternalError) as tferr:
            LOGGING.LOGGER.debug(
                'DLModelReporter could not compute predictions '
                f'on {len(X_rf)} receptive fields using the GPU.\n '
                'Trying CPU instead ...'
            )
            with tf.device("cpu:0"):
                zhat = mh.compiled.predict(X_rf, batch_size=mh.batch_size)
        # TODO Rethink : Below to common logic with _predict plots_and_reports
        if isinstance(mh.arch, SpConv3DPwiseClassif):
            _X_rf = [
                rfi.compute_active_centroids(0)
                for rfi in mh.arch.pre_runnable.last_call_receptive_fields
            ]
            _F_rf = X_rf[0]
        else:
            _X_rf = X_rf[0] if isinstance(X_rf, list) else X_rf
            _F_rf = X_rf[1] if isinstance(X_rf, list) else None
        # Do the receptive fields' plots and reports
        self.do_receptive_fields_plots_and_reports(
            mh=mh, X_rf=_X_rf, zhat_rf=zhat, X=X, y=y, F_rf=_F_rf, training=True
        )

    def do_receptive_fields_plots_and_reports(
        self, mh, X_rf, zhat_rf, X=None, y=None, F_rf=None, training=False,
    ):
        """
        Do any plot and reports related to the receptive fields when handling
        a deep learning model.

        :param mh: The model handler whose receptive fields' plots and reports
            must be done.
        :type mh: :class:`.DLModelHandler`
        :param X_rf: The receptive fields such that X_rf[i] is the matrix
            of coordinates representing the points in the i-th receptive field.
        :type X_rf: :class:`np.ndarray`
        :param zhat_rf: The output from the neural network for each receptive
            field.
        :type zhat_rf: :class:`np.ndarray`
        :param X: The structure space representing the original input point
            cloud (i.e., not each receptive field). It is not always used, but
            sometimes it is necessary, e.g., to reduce labels when using a
            :class:`.SpConv3DPwiseClassif` architecture.
        :type X: :class:`np.ndarray` or None
        :param y: The expected class for each point (considering original
            points, i.e., not the receptive fields).
        :type y: :class:`np.ndarray`
        :param F_rf: The features for each receptive field such that F_rf[i] is
            the matrix of features of the i-th receptive field. It can be None.
        :type F_rf: :class:`np.ndarray` or None
        :param training: Whether the considered receptive fields are those
            used for training (True) or not (False).
        :type training: bool
        :return: Nothing at all but the plots and reports are exported to
            the corresponding files.
        """
        # Extract output paths (either pointing to files or directories)
        rf_dir, rf_dist_report_path, rf_dist_plot_path = None, None, None
        if getattr(mh.arch.pre_runnable, 'pre_processor', None) is not None:
            rf_dir = getattr(
                mh.arch.pre_runnable.pre_processor,
                'receptive_fields_dir',
                None
            ) if not training else getattr(
                mh.arch.pre_runnable.pre_processor,
                'training_receptive_fields_dir',
                None
            )
            rf_dist_report_path = getattr(
                mh.arch.pre_runnable.pre_processor,
                'receptive_fields_distribution_report_path',
                None
            ) if not training else getattr(
                mh.arch.pre_runnable.pre_processor,
                'training_receptive_fields_distribution_report_path',
                None
            )
            rf_dist_plot_path = getattr(
                mh.arch.pre_runnable.pre_processor,
                'receptive_fields_distribution_plot_path',
                None
            ) if not training else getattr(
                mh.arch.pre_runnable.pre_processor,
                'training_receptive_fields_distribution_plot_path',
                None
            )
        # Check at least one plot or report is requested
        if (
                rf_dir is None and
                rf_dist_report_path is None and
                rf_dist_plot_path is None
        ):
            return
        # Compute the predicted and expected classes for each receptive field
        if mh.prediction_reducer is not None:  # Use prediction reducer
            if isinstance(mh.arch, SpConv3DPwiseClassif):
                yhat_rf = [  # Predictions (for each receptive field)
                    mh.prediction_reducer.select(zhat_rf_i)
                    for zhat_rf_i in zhat_rf
                ]
            else:
                yhat_rf = np.array([  # Predictions (for each receptive field)
                    mh.prediction_reducer.select(zhat_rf_i)
                    for zhat_rf_i in zhat_rf
                ])
        else:  # Use default approach
            if isinstance(mh.arch, SpConv3DPwiseClassif):
                yhat_rf = [
                    np.argmax(zhat_rf_i, axis=1)
                    if len(zhat_rf_i.shape) > 1 and zhat_rf_i.shape[-1] != 1
                    else np.round(np.squeeze(zhat_rf_i))
                    for zhat_rf_i in zhat_rf
                ]
            else:
                yhat_rf = np.array([  # Predictions (for each receptive field)
                    np.argmax(zhat_rf_i, axis=1)
                    if len(zhat_rf_i.shape) > 1 and zhat_rf_i.shape[-1] != 1
                    else np.round(np.squeeze(zhat_rf_i))
                    for zhat_rf_i in zhat_rf
                ])
        # Reduced expected classes (for each receptive field)
        # TODO Rethink : Validate for HierarchicalSGPreProcessorPP
        if y is None:
            y_rf = None
        elif isinstance(mh.arch, SpConv3DPwiseClassif):
            y_rf = mh.arch.pre_runnable.pre_processor.reduce_labels(X, y)
        else:
            y_rf = mh.arch.pre_runnable.pre_processor.reduce_labels(X_rf, y)
        if isinstance(y_rf, list):
            if len(y_rf[0].shape) > 1:
                y_rf = [np.squeeze(y_rfi) for y_rfi in y_rf]
        # Report receptive fields, if requested
        if rf_dir is not None:
            ReceptiveFieldsReport(
                X_rf=X_rf,  # X (for each receptive field)
                F_rf=F_rf,  # F (for each receptive field
                zhat_rf=zhat_rf,  # Softmax scores (for each receptive field)
                yhat_rf=yhat_rf,  # Predictions (for each receptive field)
                y_rf=y_rf,  # Expected (for each receptive field, can be None)
                class_names=mh.class_names,
                fnames=mh.arch.fnames
            ).to_file(rf_dir, mh.out_prefix)
        # Report receptive fields distribution, if requested
        if rf_dist_report_path:
            ReceptiveFieldsDistributionReport(
                yhat_rf=yhat_rf,  # Predictions (for each receptive field)
                y_rf=y_rf,  # Expected (for each receptive field, can be None)
                class_names=mh.class_names
            ).to_file(rf_dist_report_path, mh.out_prefix)
        # Plot receptive fields distribution, if requested
        if rf_dist_plot_path:
            ReceptiveFieldsDistributionPlot(
                yhat_rf=yhat_rf,  # Predictions (for each receptive field)
                y_rf=y_rf,  # Expected (for each receptive field, can be None)
                class_names=mh.class_names,
                path=rf_dist_plot_path
            ).plot(out_prefix=mh.out_prefix, logging=True)
