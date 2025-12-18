# ---   IMPORTS   --- #
# ------------------- #
from src.plot.mpl_plot import MplPlot
from src.plot.plot import PlotException
from src.plot.classification_plot import ClassificationPlot
import src.main.main_logger as LOGGING
from matplotlib import pyplot as plt
from matplotlib import patheffects
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class AdvancedClassificationPlot(MplPlot):
    """
    :author: Alberto M. Esmoris Pena

    Class to plot the advanced evaluation of a classification task.

    See :class:`.MplPlot` and :class:`.AdvancedClassificationEvaluation`.

    :ivar evals: The many evaluations on the classification based on the
        requested filters.
    :vartype evals: list of :class:`.ClassificationEvaluation`
    :ivar class_names: The names for each class involved in the classification.
    :vartype class_names: list of str
    :ivar domain_name: See :class:`.AdvancedClassificationEvaluator`.
    :ivar num_points: See :class:`.AdvancedClassificationEvaluation`.
    :ivar num_fpoints: See :class:`.AdvancedClassificationEvaluation`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of AdvancedClassificationPlot.

        :param kwargs: The key-word arguments defining the plot's attributes.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Initialize attributes of AdvancedClassificationPlot
        self.evals = kwargs.get('evals', None)
        self.domain_name = kwargs.get('domain_name', 'x')
        self.num_points = kwargs.get('num_points', None)
        self.num_fpoints = kwargs.get('num_fpoints', None)
        self.class_path = kwargs.get('class_path')
        self.confusion_matrices_path = kwargs.get('confusion_matrices_path')
        self.class_distribution_path = kwargs.get('class_distribution_path')
        # Validate evals
        if self.evals is None:
            raise PlotException(
                'AdvancedClassificationPlot failed because no evaluations '
                'were given.'
            )
        # Extract class names
        self.class_names = self.evals[0]['eval'].class_names
        # Handle serial class names when None are given
        if self.class_names is None:
            yhat_count = self.evals[0]['eval'].yhat_count
            self.class_names = [f'Class{i}' for i in range(len(yhat_count))]
        # Extract ignore classes
        self.ignore_classes = self.evals[0]['eval'].ignore_classes

    # ---   PLOT METHODS   --- #
    # ------------------------ #
    def plot(self, **kwargs):
        """
        Plot the global metrics, class-wise metrics, confusion matrices,
        and class distributions if the information is available.

        See :meth:`plot.Plot.plot`.
        """
        # Prepare path expansion, if necessary
        prefix = kwargs.get('out_prefix', None)
        if prefix is not None:
            prefix = prefix[:-1]  # Ignore '*' at the end
        # Handle global metrics plot
        if(
            self.has_global_metrics() and
            self.path is not None
        ):
            self.plot_global_metrics(**kwargs)
            path = self.path
            if prefix is not None:
                path = prefix + path[1:]
            LOGGING.LOGGER.info(
                f'AdvancedClassificationPlot wrote global metrics to "{path}"'
            )
        else:
            LOGGING.LOGGER.debug(
                'AdvancedClassificationPlot did NOT plot the global metrics.'
            )
        # Handle class-wise metrics plot
        if(
            self.has_classwise_metrics() and
            self.class_path is not None
        ):
            self.plot_classwise_metrics(**kwargs)
            path = self.class_path
            if prefix is not None:
                path = prefix + path[1:]
            LOGGING.LOGGER.info(
                'AdvancedClassificationPlot wrote class-wise metrics to '
                f'"{path}"'
            )
        else:
            LOGGING.LOGGER.debug(
                'AdvancedClassificationPlot did NOT plot the class-wise '
                'metrics.'
            )
        # Handle confusion matrices plot
        if(
            self.has_confusion_matrices() and
            self.confusion_matrices_path is not None
        ):
            self.plot_confusion_matrices(**kwargs)
            path = self.path
            if prefix is not None:
                path = prefix + path[1:]
            LOGGING.LOGGER.info(
                'AdvancedClassificationPlot wrote confusion matrices to '   
                f'"{path}"'
            )
        else:
            LOGGING.LOGGER.debug(
                'AdvancedClassificationPlot did NOT plot the confusion '
                'matrices.'
            )
        # Handle class distribution plot
        if (
            self.has_class_distribution() and
            self.class_distribution_path is not None
        ):
            self.plot_class_distribution(**kwargs)
            path = self.class_distribution_path
            if prefix is not None:
                path = prefix + path[1:]
            LOGGING.LOGGER.info(
                f'AdvancedClassificationPlot wrote class distribution to '
                f'"{path}"'
            )
        else:
            LOGGING.LOGGER.debug(
                'AdvancedClassificationPlot did NOT plot the class '
                'distribution.'
            )

    def plot_global_metrics(self, **kwargs):
        """
        Plot the global metrics.
        """
        # Build figure
        fig = plt.figure(figsize=(12, 6))
        # Extract global metrics and domain
        x = np.array([eval['x'] for eval in self.evals])
        names = self.evals[0]['eval'].metric_names
        scores = np.array([eval['eval'].metric_scores for eval in self.evals])
        # Sort global metrics by domain
        S_indices = np.argsort(x)
        x = x[S_indices]
        scores = scores[S_indices]
        num_fpoints = self.num_fpoints[S_indices]
        # Plot global metrics
        ax = fig.add_subplot(1, 2, 1)
        for i in range(len(names)):
            ax.plot(x, scores[:, i], lw=2, label=names[i])
        ax.set_xlabel(self.domain_name, fontsize=14)
        ax.set_ylabel('Global score', fontsize=14)
        ax.tick_params(axis='both', which='both', labelsize=12)
        ax.grid('both')
        ax.set_axisbelow(True)
        ax.legend(loc='best', fontsize=12)
        # Plot number of points
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, num_fpoints, color='black', lw=2, label='Number of points')
        ax.axhline(
            self.num_points, lw=2, color='tab:red', label='Total points'
        )
        ax.set_xlabel(self.domain_name, fontsize=14)
        ax.set_ylabel('Number of points', fontsize=14)
        ax.tick_params(axis='both', which='both', labelsize=12)
        ax.grid('both')
        ax.set_axisbelow(True)
        ax.legend(loc='best', fontsize=12)
        # Format figure
        fig.suptitle(
            f'Point-wise global metrics over {self.domain_name}',
            fontsize=14
        )
        fig.tight_layout()
        # Make plot effective
        self.save_show_and_clear(out_prefix=kwargs.get('out_prefix', None))

    def plot_classwise_metrics(self, **kwargs):
        """
        Plot the class-wise metrics.
        """
        # Extract class-wise metrics and domain
        x = np.array([eval['x'] for eval in self.evals])
        names = self.evals[0]['eval'].class_metric_names
        cw_scores = np.array([
            eval['eval'].class_metric_scores for eval in self.evals
        ])
        # Sort global metrics by domain
        S_indices = np.argsort(x)
        x = x[S_indices]
        cw_scores = cw_scores[S_indices]
        # Compute number of rows and columns
        n_classes = len(self.class_names)
        if n_classes == 2:
            n_cols = 2
            n_rows = 1
            row_size = 10
            col_size = 5
        else:
            n_cols = int(np.sqrt(n_classes))
            n_rows = int(np.ceil(n_classes/n_cols))
            row_size = n_cols*5
            col_size = n_rows*4
        # Build figure
        fig = plt.figure(figsize=(row_size, col_size))
        # One subplot per class
        for cidx in range(n_classes):
            ax = fig.add_subplot(n_rows, n_cols, cidx+1)
            for sidx in range(cw_scores.shape[1]):
                ax.plot(x, cw_scores[:, sidx, cidx], lw=2, label=names[sidx])
            ax.set_xlabel(self.domain_name, fontsize=14)
            ax.set_ylabel(f'{self.class_names[cidx]} score', fontsize=14)
            ax.tick_params(axis='both', which='both', labelsize=12)
            ax.grid('both')
            ax.set_axisbelow(True)
            ax.legend(loc='best', fontsize=12)
        # Format figure
        fig.suptitle(
            f'Point-wise class-wise metrics over {self.domain_name}',
            fontsize=14
        )
        fig.tight_layout()
        # Make plot effective
        path = self.path  # Store global metrics plot path
        self.path = self.class_path  # Replace path temporary
        self.save_show_and_clear(out_prefix=kwargs.get('out_prefix', None))
        self.path = path  # Restore global metrics plot path

    def plot_confusion_matrices(self, **kwargs):
        """
        Plot the confusion matrices.
        """
        # Compute number of rows and columns
        n_evals = len(self.evals)
        class_names = list(self.class_names)
        n_classes = len(class_names)
        if n_evals == 2:
            n_cols = 2
            n_rows = 1
            row_size = 5
            col_size = 10
        else:
            n_cols = int(np.sqrt(n_evals))
            n_rows = int(np.ceil(n_evals/n_cols))
            row_size = n_rows*max(5, 1+n_classes)
            col_size = n_cols*max(5, 1+n_classes)
        # Build figure
        fig = plt.figure(figsize=(row_size, col_size))
        # Do one confusion matrix for each subplot
        norm_type = '#ERROR#'
        for eidx, eval in enumerate(self.evals):
            x = eval['x']
            eval = eval['eval']
            conf_mat = eval.conf_mat
            ax = fig.add_subplot(n_rows, n_cols, eidx+1)
            norm_type = ClassificationPlot.do_confusion_matrix_plot(
                fig, ax, conf_mat, class_names,
                ignore_classes=self.ignore_classes,
                normalization_strategy=eval.conf_mat_norm_type,
                title=f'{self.domain_name} = {x:.3f}'
            )
        # Format figure
        fig.suptitle(
            f'Confusion matrices over {self.domain_name} with {norm_type}',
            fontsize=14
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        # Make plot effective
        path = self.path  # Store global metrics plot path
        self.path = self.confusion_matrices_path  # Replace path temporary
        self.save_show_and_clear(out_prefix=kwargs.get('out_prefix', None))
        self.path = path  # Restore global metrics plot path

    def plot_class_distribution(self, **kwargs):
        """
        Plot the class distribution for each node in the domain.
        """
        # Compute number of rows and columns
        n_evals = len(self.evals)
        n_plots = n_evals*2
        if n_plots == 2:
            n_cols = 2
            n_rows = 1
            row_size = 7
            col_size = 12
        else:
            n_cols = int(np.sqrt(n_plots))
            if n_cols % 2 != 0:
                n_cols += 1
            n_rows = int(np.ceil(n_plots/n_cols))
            row_size = 2+n_cols*5
            col_size = 1+n_rows*5
        # Build figure
        fig = plt.figure(figsize=(row_size, col_size))
        # Do two bar subplots per node in the domain
        for eidx, eval in enumerate(self.evals):
            # Extract values of interest from evaluation
            x = eval['x']
            eval = eval['eval']
            # Plot predictions distribution
            ax_pred = fig.add_subplot(n_rows, n_cols, 1+eidx*2)
            ax_ref = fig.add_subplot(n_rows, n_cols, 2+eidx*2)
            ClassificationPlot.do_class_distributions_subplots(
                ax_pred, ax_ref, eval.yhat_count, eval.y_count,
                self.class_names,
                pred_title=f'Reference on {self.domain_name} = {x:.3}',
                ref_title=f'Reference on {self.domain_name} = {x:.3}'
            )
        # Format figure
        fig.suptitle(
            f'Point-wise class distributions over {self.domain_name}',
            fontsize=14
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        # Make plot effective
        path = self.path  # Store global metrics plot path
        self.path = self.class_distribution_path  # Replace path temporary
        self.save_show_and_clear(out_prefix=kwargs.get('out_prefix', None))
        self.path = path  # Restore global metrics plot path

    # ---  CHECK METHODS  --- #
    # ----------------------- #
    def has_global_metrics(self):
        """
        Check whether the plot contains global metrics.

        :return: True if the plot contains global metrics, False otherwise.
        """
        return (
            self.evals[0]['eval'].metric_names is not None and
            self.evals[0]['eval'].metric_scores is not None
        )

    def has_classwise_metrics(self):
        """
        Check whether the plot contains class-wise metrics.

        :return: True if the plot contains class-wise metrics, False otherwise.
        """
        return (
            self.class_names is not None and
            self.evals[0]['eval'].class_metric_names is not None and
            self.evals[0]['eval'].class_metric_scores is not None
        )

    def has_confusion_matrices(self):
        """
        Check whether the plot contains confusion matrices.

        :return: True if the plot contains confusion matrices, False otherwise.
        """
        return (
            self.evals[0]['eval'].conf_mat is not None and
            self.class_names is not None
        )

    def has_class_distribution(self):
        """
        Check whether the plot contains all the information needed to plot the
        class distribution.

        :return: True if the plot contains all the information needed to plot
            the class distribution.
        """
        return (
            self.class_names is not None and
            self.evals[0]['eval'].yhat_count is not None and
            self.evals[0]['eval'].y_count is not None
        )
