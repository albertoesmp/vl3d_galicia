# ---   IMPORTS   --- #
# ------------------- #
from src.eval.evaluation import Evaluation, EvaluationException
from src.report.advanced_classification_report import \
    AdvancedClassificationReport
from src.plot.advanced_classification_plot import AdvancedClassificationPlot


# ---   CLASS   --- #
# ----------------- #
class AdvancedClassificationEvaluation(Evaluation):
    """
    :author: Alberto M. Esmoris Pena

    Class representing the result of evaluating a classification in and
    advanced way. See :class:`.AdvancedClassificationEvaluator`.

    :ivar evals: The evaluation of the classification for each filter.
    :vartype evals: list of :class:`.ClassificationEvaluation`
    :ivar domain_name: See :class:`.AdvancedClassificationEvaluator`.
    :ivar num_points: The number of points without filtering (i.e., the
        original number of points).
    :vartype num_points: int
    :ivar num_fpoints: The number of points for each evaluation (i.e., the
        points preserved after applying each filter).
    :vartype num_fpoints: :class:`np.ndarray` of int
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate an AdvancedClassificationEvaluation.

        :param kwargs: The attributes for the AdvancedClassificationEvaluation.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Initialize attributes of the AdvancedClassificationEvaluation
        self.evals = kwargs.get('evals', None)
        self.domain_name = kwargs.get('domain_name', 'x')
        self.num_points = kwargs.get('num_points', None)
        self.num_fpoints = kwargs.get('num_fpoints', None)
        # Validate attributes
        if self.evals is None or len(self.evals) < 1:
            raise EvaluationException(
                'AdvancedEvaluationClassification cannot be built without '
                'evaluations.'
            )
        if self.num_points < 1:
            raise EvaluationException(
                'AdvancedClassificationEvaluation cannot be built for less '
                'than one point.'
            )
        if self.num_fpoints is None or len(self.num_fpoints) < 1:
            raise EvaluationException(
                'AdvancedClassificationEvaluation cannot be built for less '
                'than one filtered points.'
            )

    # ---   EVALUATION METHODS   --- #
    # ------------------------------ #
    def report(self, **kwargs):
        """
        Transform the AdvancedClassificationEvaluation into an
        :class:`.AdvancedClassificationReport`.

        See :class:`.AdvancedClassificationReport`.

        :return: The :class:`.AdvancedClassificationReport` representing the
            :class:`.AdvancedClassificationEvaluation`.
        :rtype: :class:`.AdvancedClassificationReport`
        """
        return AdvancedClassificationReport(
            evals=self.evals,
            domain_name=self.domain_name,
            num_points=self.num_points,
            num_fpoints=self.num_fpoints
        )

    def can_report(self):
        """
        See :class:`.Evaluation` and :meth:`.Evaluation.can_report`.
        """
        return self.evals[0]['eval'].can_report()

    def plot(self, **kwargs):
        """
        Transform the :class:`.AdvancedClassificationEvaluator` into a
        :class:`.AdvancedClassificationPlot`.

        Se :class:`.AdvancedClassificationPlot`.

        :param kwargs: The key-word arguments for the plot.
        :return: The :class:`.AdvancedClassificationPlot` representing the
            :class:`.ClassificationEvaluation`.
        :rtype: :class:`.AdvancedClassificationPlot`
        """
        return AdvancedClassificationPlot(
            evals=self.evals,
            domain_name=self.domain_name,
            num_points=self.num_points,
            num_fpoints=self.num_fpoints,
            path=kwargs.get('path', None),
            class_path=kwargs.get('class_path', None),
            confusion_matrices_path=kwargs.get(
                'confusion_matrices_path', None
            ),
            class_distribution_path=kwargs.get(
                'class_distribution_path', None
            ),
            show=kwargs.get('show', False)
        )

    def can_plot(self):
        """
        See :class:`.Evaluation` and :meth:`.Evaluation.can_plot`.
        """
        return self.evals[0]['eval'].can_plot()
