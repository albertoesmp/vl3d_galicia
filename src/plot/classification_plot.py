# ---   IMPORTS   --- #
# ------------------- #
from src.plot.mpl_plot import MplPlot
import src.main.main_logger as LOGGING
from src.main.main_config import VL3DCFG
from matplotlib import pyplot as plt
from matplotlib import patheffects
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class ClassificationPlot(MplPlot):
    """
    :author: Alberto M. Esmoris Pena

    Class to plot the evaluation of a classification task.

    See :class:`.MplPlot` and :class:`.ClassificationEvaluation`.

    :ivar class_names: See :class:`.ClassificationEvaluation`.
    :ivar ignore_classes: See :class:`.ClassificationEvaluation`.
    :ivar yhat_count: See :class:`.ClassificationEvaluation`.
    :ivar y_count: See :class:`.ClassificationEvaluation`.
    :ivar conf_mat: See :class:`.ClassificationEvaluation`.
    :ivar class_distribution_path: The path where the class distribution plot
        must be written. Can be None. In that case, no class distribution
        plot will be written.
    :vartype class_distribution_path: str
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of ClassificationPlot.

        :param kwargs: The key-word arguments defining the plot's attributes.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Initialize attributes of ClassificationPlot
        self.class_names = kwargs.get('class_names', None)
        self.ignore_classes = kwargs.get('ignore_classes', None)
        self.yhat_count = kwargs.get('yhat_count', None)
        self.y_count = kwargs.get('y_count', None)
        self.conf_mat = kwargs.get('conf_mat', None)
        self.conf_mat_norm_type = kwargs.get('conf_mat_norm_type', None)
        self.class_distribution_path = kwargs.get('class_distribution_path')

    # ---   PLOT METHODS   --- #
    # ------------------------ #
    def plot(self, **kwargs):
        """
        Plot the confusion matrix and the class distribution if the information
        is available.

        See :meth:`plot.Plot.plot`.
        """
        # Prepare path expansion, if necessary
        prefix = kwargs.get('out_prefix', None)
        if prefix is not None:
            prefix = prefix[:-1]  # Ignore '*' at the end
        # Handle confusion matrix plot
        if self.has_confusion_matrix() and self.path is not None:
            self.plot_confusion_matrix(**kwargs)
            path = self.path
            if prefix is not None:
                path = prefix + path[1:]
            LOGGING.LOGGER.info(
                f'ClassificationPlot wrote confusion matrix to "{path}"'
            )
        else:
            LOGGING.LOGGER.debug(
                'ClassificationPlot did NOT plot the confusion matrix.'
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
                f'ClassificationPlot wrote class distribution to "{path}"'
            )
        else:
            LOGGING.LOGGER.debug(
                'ClassificationPlot did NOT plot the class distribution.'
            )

    def plot_confusion_matrix(self, **kwargs):
        """
        Plot the confusion matrix.
        """
        # Build figure
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(1, 1, 1)
        # Plot confusion matrix
        conf_mat = self.conf_mat
        class_names = list(self.class_names)
        ClassificationPlot.do_confusion_matrix_plot(
            fig, ax, conf_mat, class_names,
            ignore_classes=self.ignore_classes,
            normalization_strategy=self.conf_mat_norm_type,
            title=None
        )
        # Format figure
        fig.tight_layout()
        # Make plot effective
        self.save_show_and_clear(out_prefix=kwargs.get('out_prefix', None))

    def plot_class_distribution(self, **kwargs):
        """
        Plot the class distribution.
        """
        # Build figure
        fig = plt.figure(figsize=(14, 5))
        # Plot predictions and reference distributions
        ax_pred = fig.add_subplot(1, 2, 1)
        ax_ref = fig.add_subplot(1, 2, 2)
        ClassificationPlot.do_class_distributions_subplots(
            ax_pred, ax_ref, self.yhat_count, self.y_count, self.class_names
        )
        # Format figure
        fig.suptitle('Point-wise class distributions', fontsize=14)
        fig.tight_layout()
        # Make plot effective
        path = self.path  # Store confusion matrix plot path
        self.path = self.class_distribution_path  # Replace path temporary
        self.save_show_and_clear(out_prefix=kwargs.get('out_prefix', None))
        self.path = path  # Restore confusion matrix plot path

    # ---   PLOT UTILS   --- #
    # ---------------------- #
    @staticmethod
    def do_class_distributions_subplots(
        ax_pred, ax_ref, yhat_count, y_count, class_names,
        pred_title='Predictions', ref_title='Reference'
    ):
        """
        Handle two bar subplots, one for the distribution of predictions
        on the different classes, the other for the distribution of the
        references.

        :param ax_pred: The axes where the distribution of predictions will
            be plotted.
        :param ax_ref: The axes where the distribution of references will be
            plotted.
        :param yhat_count: See :class:`.ClassificationEvaluation`.
        :param y_count: See :class:`.ClassificationEvaluation`.
        :param class_names: See :class:`.ClassificationEvaluator`.
        :param pred_title: The title for the predictions-based subplot.
        :type pred_title: str
        :param ref_title: The title for the reference-based subplot.
        :type ref_title: str
        """
        # Find positions for the class names
        x = np.arange(len(yhat_count))
        # Plot predictions distribution
        ax_pred.set_title(pred_title, fontsize=12)
        ax_pred.bar(
            x, yhat_count,
            tick_label=class_names, linewidth=1,
            edgecolor='black', color='tab:blue'
        )
        axes = [ax_pred]
        # Plot references distribution
        ax_ref.set_title(ref_title, fontsize=12)
        ax_ref.bar(
            x, y_count,
            tick_label=class_names, linewidth=1,
            edgecolor='black', color='tab:green'
        )
        axes.append(ax_ref)
        # Format axes
        for ax in axes:
            ax.tick_params(axis='both', which='both', labelsize=12)
            ax.tick_params(axis='x', which='both', labelrotation=70)
            ax.set_ylabel('Number of points', fontsize=12)
            ax.grid('both')
            ax.set_axisbelow(True)

    @staticmethod
    def do_confusion_matrix_plot(
        fig, ax, cmat, class_names,
        ignore_classes=None,
        normalization_strategy=None,
        title=None
    ):
        """
        Handle the plot of the confusion matrix on the given figure and axes.

        :param fig: The figure where the confusion matrix must be plotted.
        :param ax: The axes (inside the figure) where the confusion matrix must
            be plotted.
        :param cmat: The confusion matrix to be plotted.
        :type cmat: :class:`np.ndarray`
        :param class_names: See :class:`.ClassificationEvaluator`.
        :param ignore_classes: See :class:`.ClassificationEvaluator`.
        :param normalization_strategy: See :class:`.ClassificationEvaluation`,
            more concretely, the `conf_mat_norm_type` attribute (ivar).
        :param title: Optional title to be assigned to the axes where the
            confusion matrix is represented.
        :type title: str or None
        """
        # Handle normalization
        norm_type = 'no normalization'
        if normalization_strategy is not None:
            # Row (by reference) normalization
            if normalization_strategy.lower() == 'row':
                cmat = (cmat.T / np.sum(cmat, axis=1))
                norm_type = 'row-wise normalization'
            # Col (by prediction) normalization
            elif normalization_strategy.lower() in ['col', 'column']:
                cmat = cmat / np.sum(cmat, axis=0)
                norm_type = 'column-wise normalization'
            # Full normalization
            elif normalization_strategy.lower in ['full', 'all']:
                cmat = cmat / np.sum(cmat)
                norm_type = 'full normalization'
        # Handle title
        if title is not None:
            ax.set_title(title, fontsize=12)
        # Handle unexpected number of classes / dimensionalities
        n_classes = len(class_names)
        if n_classes > len(cmat):  # More classes than expected
            cmat = np.pad(
                cmat,
                pad_width=[0, n_classes-cmat.shape[0]]
            )
        elif n_classes < len(cmat):  # Handle ignored classes
            for i in range(n_classes, len(cmat)):
                class_names.append(ignore_classes[i-n_classes])
        # Plot the matrix
        cmat_format= VL3DCFG['EVAL']['ClassificationEvaluator'][
            'confusion_matrix_format'
        ]
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cmat,
            display_labels=class_names,
        )
        disp.plot(
            ax=ax,
            cmap=cmat_format['cmap'],
            values_format=cmat_format['values_format'],
            text_kw={
                'fontsize': cmat_format['fontsize'],
                'fontweight': cmat_format['fontweight'],
                'color': cmat_format['fontcolor'],
                'path_effects': [
                    patheffects.withStroke(
                        linewidth=cmat_format['stroke_width'],
                        foreground=cmat_format['stroke_foreground']
                    )
                ]
            },
            colorbar=False
        )
        # Color bar
        cbar = fig.colorbar(disp.im_)
        cbar.ax.tick_params(labelsize=cmat_format['colorbar_labelsize'])
        # Format axes
        ax.tick_params(
            axis='both',
            which='both',
            labelsize=cmat_format['tick_labelsize'],
            length=cmat_format['tick_length'],
            width=cmat_format['tick_width']
        )
        ax.tick_params(
            axis='x', which='both',
            labelrotation=cmat_format['xlabel_rotation']
        )
        ax.tick_params(
            axis='y', which='both',
            labelrotation=cmat_format['ylabel_rotation']
        )
        ax.set_xlabel(ax.get_xlabel(), fontsize=cmat_format['xlabel_size'])
        ax.set_ylabel(ax.get_ylabel(), fontsize=cmat_format['ylabel_size'])
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(  # Plot border size
                cmat_format['border_size']
            )
        # Return type of applied normalization
        return norm_type

    # ---  CHECK METHODS  --- #
    # ----------------------- #
    def has_confusion_matrix(self):
        """
        Check whether the plot contains a confusion matrix.

        :return: True if the plot contains a confusion matrix, False otherwise.
        """
        return self.conf_mat is not None and self.class_names is not None

    def has_class_distribution(self):
        """
        Check whether the plot contains all the information needed to plot the
        class distribution.

        :return: True if the plot contains all the information needed to plot
            the class distribution.
        """
        return (
            self.class_names is not None and
            self.yhat_count is not None and
            self.y_count is not None
        )
