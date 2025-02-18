# ---   IMPORTS   --- #
# ------------------- #
from src.report.report import Report, ReportException
from src.inout.io_utils import IOUtils
import src.main.main_logger as LOGGING
import numpy as np
import os


# ---   CLASS   --- #
# ----------------- #
class AdvancedClassificationReport(Report):
    """
    :author: Alberto M. Esmoris PEna

    Class to handle advanced reports related to classifications.
    See :class:`.Report`, :class:`.ClassificationModel`,
    :class:`.AdvancedClassificationEvaluator`, and
    :class:`.AdvancedClassificationEvaluation`.

    :ivar evals: The many evaluations on the classification based on the
        requested filters.
    :vartype evals: list of :class:`.ClassificationEvaluation`
    :ivar class_names: The names for each class involved in the classification.
    :vartype class_names: list of str
    :ivar domain_name: See :class:`.AdvancedClassificationEvaluator`.
    :ivar num_points: See :class:`.AdvancedClassificationEvaluation`.
    :ivar num_fpoints: See :class:`.AdvancedClassificationEvaluation`.
    """
    # ---  STATIC CLASS ATTRIBUTES  --- #
    # --------------------------------- #
    CONFMAT_SEPARATOR='----------------'

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of AdvancedClassificationReport.

        :param kwargs: The key-word arguments defining the report's attributes.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Basic attributes of the AdvancedClassificationReport
        self.evals = kwargs.get('evals', None)
        self.domain_name = kwargs.get('domain_name', 'x')
        self.num_points = kwargs.get('num_points', None)
        self.num_fpoints = kwargs.get('num_fpoints', None)
        # Validate evals
        if self.evals is None:
            raise ReportException(
                'AdvancedClassificationReport failed because no evaluations '
                'were given.'
            )
        # Extract class names
        self.class_names = self.evals[0]['eval'].class_names
        # Handle serial class names when None are given
        if self.class_names is None:
            yhat_count = self.evals[0]['eval'].yhat_count
            self.class_names = [f'Class{i}' for i in range(len(yhat_count))]

    # ---   TO STRING   --- #
    # --------------------- #
    def __str__(self):
        """
        The string representation of the advanced report about a
        classification.
        See :class:`.Report` and also :meth:`.Report.__str__`.
        """
        # Initialize
        s = '\n    Advanced classification report\n' \
            '=========================================\n'
        # Report number of points withotu filters
        s += f'\nOriginal number of points: {self.num_points}\n'
        # Fill with available information
        if self.has_global_eval_info():
            s += '\n'+self.to_global_eval_string()+'\n'
        if self.has_class_eval_info():
            s += '\n'+self.to_class_eval_string()+'\n'
        if self.has_confusion_matrix():
            s += '\n'+self.to_confusion_matrices_string()+'\n'
        if self.has_class_distribution_info():
            s += '\n'+self.to_class_distribution_string()+'\n'
        # Return
        return s

    def to_global_eval_string(self):
        """
        Generate the string representing the advanced classification report
        with respect to the advanced global evaluation.

        :return: String representing the advanced classification report with
            respect to the advanced global evaluation.
        """
        # --- Introduction --- #
        s = 'Advanced global classification evaluation:\n'
        # ---  Head  --- #
        s += f'{self.domain_name:>16.16},      num. points,'
        for mname in self.evals[0]['eval'].metric_names:
            s += f'{mname:>11.11},'
        s = s[:-1] + '\n'
        # ---  Body  --- #
        for i, eval in enumerate(self.evals):
            x = eval['x']
            eval = eval['eval']
            s += f'  {x:16.3f},'
            s += f' {self.num_fpoints[i]:16d},'
            for score in eval.metric_scores:
                s += f' {100*score:10.3f},'
            s = s[:-1] + '\n'
        # Return
        return s

    def to_class_eval_string(self):
        """
        Generate the string representing the advanced classification report
        with respect to the advanced class-wise evaluation.

        :return: String representing the advanced classification report with
            respect to the advanced class-wise evaluation.
        """
        # --- Introduction --- #
        s = 'Advanced class-wise classification evaluation:\n'
        # ---  Head  --- #
        s += 'class,            '
        s += f'{self.domain_name:>16.16},'
        for mname in self.evals[0]['eval'].class_metric_names:
            s += f'{mname:>11.11},'
        s  = s[:-1]
        # ---  Body  --- #
        for eval in self.evals:
            x = eval['x']
            eval = eval['eval']
            for i, class_name in enumerate(self.class_names):
                s += f'\n{class_name:16.16},'
                s += f' {x:16.3f},'
                for class_score in eval.class_metric_scores[:, i]:
                    s += f' {100*class_score:10.3f},'
                s = s[:-1]
        s += '\n'
        # Return
        return s

    def to_confusion_matrices_string(self):
        """
        Generate the string representing the advanced classification report
            with respect to the advanced confusion matrices.

        :return: String representing the advanced classification report with
            respect to the advanced confusion matrices.
        """
        # --- Introduction --- #
        s = 'Advanced confusion matrices (rows are true labels, columns are predictions):'
        # ---  Matrix  --- #
        for eval in self.evals:
            x = eval['x']
            eval = eval['eval']
            nrows, ncols = eval.conf_mat.shape
            s += f'\nConfusion matrix ({self.domain_name} = {x:.3f}):\n'
            s += f'{self.CONFMAT_SEPARATOR}\n'
            for i in range(nrows):
                for j in range(ncols):
                    s += f'{eval.conf_mat[i, j]:9}, '
                s = s[:-2] + '\n'
            s += f'{self.CONFMAT_SEPARATOR}\n'
        # Return
        return s

    def to_class_distribution_string(self):
        """
        Generate the string representing the advanced classification report
        with respect to the advanced class distribution.

        :return: String representing the advanced classification report with
            respect to the advanced class distribution.
        """
        # --- Introduction --- #
        s = 'Advanced class distribution:\n'
        # ---  Head  --- #
        s += f'CLASS           , {self.domain_name:>16.16},    PRED. COUNT, '\
             'PRED. PERCENT.,     TRUE COUNT,  TRUE PERCENT.\n'
        # ---  Body  --- #
        for eval in self.evals:
            x = eval['x']
            eval = eval['eval']
            yhat_percentage = 100 * eval.yhat_count / np.sum(eval.yhat_count)
            y_percentage = 100 * eval.y_count / np.sum(eval.y_count)
            for i in range(len(self.class_names)):
                s += f'{self.class_names[i]:16.16}, ' \
                     f'{x:16.3f}, ' \
                     f'{eval.yhat_count[i]:14}, ' \
                     f'{yhat_percentage[i]:14.3f}, ' \
                     f'{eval.y_count[i]:14}, ' \
                     f'{y_percentage[i]:14.3f}\n'
        # Return
        return s

    # ---   TO FILE   --- #
    # ------------------- #
    def to_file(
        self,
        report_path=None,
        class_report_path=None,
        confusion_matrix_report_path=None,
        class_distribution_report_path=None,
        out_prefix=None
    ):
        """
        Write the advanced classification report to files.

        :param report_path: See :class:`.ClassificationEvaluator`.
        :param class_report_path: See :class:`.ClassificationEvaluator`.
        :param confusion_matrix_report_path: See
            :class:`.ClassificationEvaluator`.
        :param class_distribution_report_path: See
            :class:`.ClassificationEvaluator`.
        :param out_prefix: The output prefix to expand the path (OPTIONAL).
        :type out_prefix: str
        :return: Nothing, the output is written to a file.
        """
        # Prepare many reports
        report_names = [
            'Advanced global evaluation',
            'Advanced class evaluation',
            'Advanced confusion matrices',
            'Advanced class distribution'
        ]
        paths = [
            report_path,
            class_report_path,
            confusion_matrix_report_path,
            class_distribution_report_path
        ]
        checks = [
            self.has_global_eval_info,
            self.has_class_eval_info,
            self.has_confusion_matrix,
            self.has_class_distribution_info
        ]
        to_strings = [
            self.to_global_eval_string,
            self.to_class_eval_string,
            self.to_confusion_matrices_string,
            self.to_class_distribution_string
        ]

        # Do many reports
        # TODO Rethink : Abstract to common logic with ClassificationReport ?
        for i in range(len(paths)):
            # Extract iteration variables
            report_name = report_names[i]
            path = paths[i]
            if path is None:
                LOGGING.LOGGER.debug(
                    f'AdvancedClassificationReport skips "{report_name}" '
                    'report because no path was given.'
                )
                continue
            check = checks[i]
            to_string = to_strings[i]
            # Check info
            if not check():
                LOGGING.LOGGER.debug(
                    'AdvancedClassificationReport did NOT write report on '
                    f'{report_name} to "{path}" because data was not '
                    'available.'
                )
                continue
            # Expand path if necessary
            if out_prefix is not None and path[0] == '*':
                path = out_prefix[:-1] + path[1:]
            # Check path
            IOUtils.validate_path_to_directory(
                os.path.dirname(path),
                'Cannot find the directory to write the report:'
            )
            # Write
            with open(path, 'w') as outf:
                outf.write(to_string())
            # Log
            LOGGING.LOGGER.info(f'Report on {report_name} written to "{path}"')

    # ---  CHECK METHODS  --- #
    # ----------------------- #
    def has_global_eval_info(self):
        """
        Check whether the report contains information about the global
        evaluation.

        :return: True if the report contains information about the global
            evaluation, False otherwise.
        """
        return (
            self.evals[0]['eval'].metric_scores is not None and
            self.evals[0]['eval'].metric_names is not None
        )

    def has_class_eval_info(self):
        """
        Check whether the report contains information about the class-wise
        evaluation.

        :return: True if the report contains information about the class-wise
            evaluation, False otherwise.
        """
        return (
            self.evals[0]['eval'].class_names is not None and
            self.evals[0]['eval'].class_metric_scores is not None and
            self.evals[0]['eval'].class_metric_names is not None
        )

    def has_confusion_matrix(self):
        """
        Check whether the report contains the confusion matrix.

        :return: True if the report contains the confusion matrix, False
            otherwise.
        """
        return self.evals[0]['eval'].conf_mat is not None

    def has_class_distribution_info(self):
        """
        Check whether the report contains information about the class
        distribution.

        :return: True if the report contains information about the class
            distribution, False otherwise.
        """
        return (
            self.evals[0]['eval'].class_names is not None and
            self.evals[0]['eval'].yhat_count is not None and
            self.evals[0]['eval'].y_count is not None
        )
