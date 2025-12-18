# ---   IMPORTS   --- #
# ------------------- #
from src.eval.classification_evaluator import ClassificationEvaluator,\
    Evaluator, EvaluatorException
from src.eval.advanced_classification_evaluation import \
    AdvancedClassificationEvaluation
import src.main.main_logger as LOGGING
from src.utils.dict_utils import DictUtils
import numpy as np
import time


# ---   CLASS   --- #
# ----------------- #
class AdvancedClassificationEvaluator(Evaluator):
    r"""
    :author: Alberto M. Esmoris Pena

    Class to evaluate classification-like predictions against
    expected/reference classes in an advanced way, i.e., by computing many
    evaluations considering different filters.

    The arguments of the :class:`.AdvancedClassificationEvaluator` include
    those of the :class:`.ClassificationEvaluator`. Only those that are
    introduced by the :class:`.AdvancedClassificationEvaluator` are documented
    here. See :class:`.ClassificationEvaluator` for details on the common
    attributes.

    See also :class:`.AdvancedClassificationEvaluation`.

    :ivar evaluator: The :class:`.ClassificationEvaluator` used to compute
        the filter-wise classification evaluations.
    :vartype evaluator: :class:`.ClassificationEvaluator`.
    :ivar filters: List of filters such that one evaluation will be carried
        out for each filter. An example of filter is given below:

        .. code-block:: json

            {
                "name": "pwe0_1",
                "x": 0.1,
                "conditions": [
                    {
                        "value_name": "classification",
                        "condition_type": "not_equals",
                        "value_target": 2,
                        "action": "discard"
                    },
                    {
                        "value_name": "PointWiseEntropy",
                        "condition_type": "less_than_or_equal_to",
                        "value_target": 0.1,
                        "action": "preserve"
                    }
                ]
            }

        The filter above will filter out all the points that are classified
        in the class 2 (third class, note they start at zero) and then it
        will consider only those that have a point-wise entropy
        :math:`\leq 1/10`. The value of the `"x"` attribute will be used
        in the figures to represent nodes along the :math:`x`-axis and in the
        output CSV (report) to identify to which evaluation corresponds
        each row.
    :vartype filters: list of dict
    :ivar domain_name: The name of the variable that constitutes the domain
        of the advanced evaluation (:math:`x`-axis name).
    :vartype domain_name: str
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_eval_args(spec):
        """
        Extract the arguments to initialize/instantiate an
        :class:`.AdvancedClassificationEvaluator` from a key-word specification.

        See :meth:`.ClassificationEvaluator.extract_eval_args`.

        :param: spec The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate an
            :class:`.AdvancedClassificationEvaluator`.
        """
        # Initialize
        kwargs = ClassificationEvaluator.extract_eval_args(spec)
        # Handle AdvancedClassificationEvaluator arguments
        kwargs['plot_path'] = spec.get('plot_path', None)
        kwargs['class_plot_path'] = spec.get('class_plot_path', None)
        kwargs['domain_name'] = spec.get('domain_name', None)
        kwargs['filters'] = spec.get('filters', None)
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate an  AdvancedClassificationEvaluator.

        :param kwargs: The attributes for the AdvancedClassificationEvaluator.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Instantiate underlying classification evaluator
        self.evaluator = ClassificationEvaluator(**kwargs)
        # Assign AdvancedClassificationEvaluator attributes
        self.plot_path = kwargs.get('plot_path', None)
        self.class_plot_path = kwargs.get('class_plot_path', None)
        self.domain_name = kwargs.get('domain_name', 'x')
        self.filters = kwargs.get('filters', None)
        # Check filters are given
        if self.filters is None or len(self.filters) < 1:
            raise EvaluatorException(
                'Cannot build AdvancedClassificationEvaluator with no filters.'
            )

    # ---  EVALUATOR METHODS  --- #
    # --------------------------- #
    def eval(self, yhat, y=None, fnames=None, F=None):
        r"""
        Evaluate predicted classes (:math:`\hat{y}`) against expected/reference
        classes (:math:`y`) one time for each given filter.

        See :meth:`.ClassificationEvaluator.eval`.
        """
        start = time.perf_counter()
        # Validate arguments
        if y is None:
            raise EvaluatorException(
                'AdvancedClassificationEvaluator cannot evaluate without the '
                'expected or reference labels.'
            )
        # Compute one evaluation for each filter
        m = len(yhat)  # Number of points without filters
        fm = []  # Number of filtered points (those preserved, no filtered out)
        evals = []
        for filter in self.filters:
            x, filtered_yhat, filtered_y = self.apply_filter(
                filter, yhat, y=y, fnames=fnames, F=F
            )
            fm.append(len(filtered_yhat))
            eval = self.evaluator.eval(filtered_yhat, y=filtered_y)
            evals.append({'x': x, 'eval': eval})
        # Log execution time
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'AdvancedClassificationEvaluator evaluated {len(yhat)} points in '
            f'{end-start:.3f} seconds.'
        )
        # Return
        return AdvancedClassificationEvaluation(
            evals=evals, domain_name=self.domain_name,
            num_points=m, num_fpoints=np.array(fm)
        )

    def __call__(self, pcloud, **kwargs):
        """
        Evaluate with extra logic that is convenient for pipeline-based
        execution.

        See :meth:`.Evaluator.eval`.
        """
        # Prepare and compute evaluation
        fnames = self.get_fnames_from_filters()
        if kwargs.get('yhat', None) is not None:
            yhat = kwargs['yhat']
        else:
            yhat = pcloud.get_predictions_vector()
        y = pcloud.get_classes_vector()
        F = pcloud.get_features_matrix(fnames)
        ev = self.eval(yhat, y=y, fnames=fnames, F=F)
        # Handle computed evaluation (reports and plots)
        out_prefix = kwargs.get('out_prefix', None)
        if ev.can_report():
            report = ev.report()
            LOGGING.LOGGER.info(report.to_string())
            report_path = kwargs.get(
                'report_path', self.evaluator.report_path
            )
            class_report_path = kwargs.get(
                'class_report_path', self.evaluator.class_report_path
            )
            confusion_matrix_report_path = kwargs.get(
                'confusion_matrix_report_path',
                self.evaluator.confusion_matrix_report_path
            )
            class_distrib_report_path = kwargs.get(
                'class_distribution_report_path',
                self.evaluator.class_distribution_report_path
            )
            if(
                report_path is not None or
                class_report_path is not None or
                confusion_matrix_report_path is not None or
                class_distrib_report_path is not None
            ):
                start = time.perf_counter()
                report.to_file(
                    report_path=report_path,
                    class_report_path=class_report_path,
                    confusion_matrix_report_path=confusion_matrix_report_path,
                    class_distribution_report_path=class_distrib_report_path,
                    out_prefix=out_prefix
                )
                end = time.perf_counter()
                LOGGING.LOGGER.info(
                    'Advanced classification reports written in '
                    f'{end-start:.3f} seconds.'
                )
        if ev.can_plot():
            plot_path = kwargs.get(
                'plot_path',
                self.plot_path
            )
            class_plot_path = kwargs.get(
                'class_plot_path',
                self.class_plot_path
            )
            cmat_plot_path = kwargs.get(
                'confusion_matrix_plot_path',
                self.evaluator.confusion_matrix_plot_path
            )
            class_distribution_plot_path = kwargs.get(
                'class_distribution_plot_path',
                self.evaluator.class_distribution_plot_path
            )
            if plot_path is not None:
                start = time.perf_counter()
                ev.plot(
                    path=plot_path,
                    class_path=class_plot_path,
                    confusion_matrices_path=cmat_plot_path,
                    class_distribution_path=class_distribution_plot_path
                ).plot(out_prefix=out_prefix)
                end = time.perf_counter()
                LOGGING.LOGGER.info(
                    f'Advanced classification plots written in {end-start:.3f}'
                    'seconds.'
                )

    # ---  ADVANCED CLASSIFICATION EVALUATION METHODS  --- #
    # ---------------------------------------------------- #
    def apply_filter(self, f, yhat, y=None, fnames=None, F=None):
        """
        Apply the given filter on the predictions and reference classes to
        extract the subset of predicted classes that must be evaluated.

        :param f: The filter to be applied.
        :type f: dict
        :param yhat: The predicted classes to be filtered.
        :type yhat: :class:`np.ndarray`
        :param y: The reference classes to be filtered.
        :type y: :class:`np.ndarray`
        :param fnames: The names of the features in :math:`\pmb{F}`.
        :type fnames: list of str
        :param F: The feature space matrix representing the point cloud to be
            evaluated.
        :type F: :class:`np.ndarray`
        :return: Return the domain node (x), filtered predictions, and
            filtered reference classes as a 3-tuple.
        :rtype: tuple
        """
        # Get filter name
        name = f.get('name', '#UnnamedFilter#')
        # Get node in the domain
        x = f.get('x', None)
        if x is None:
            raise EvaluatorException(
                'AdvancedClassificationEvaluator failed to apply filter '
                f'"{name}" because no domain value was specified.'
            )
        # Apply filter conditions
        conditions = f.get('conditions', None)
        if conditions is None:
            raise EvaluatorException(
                'AdvancedClassificationEvaluator failed to apply filter '
                f'"{name}" because no conditions were given.'
            )
        mask = np.ones_like(yhat, dtype=bool)  # Preserve mask
        for cond in f['conditions']:
            self.apply_condition(
                mask, cond, yhat, y=y, fnames=fnames, F=F, name=name
            )
        fyhat, fy = yhat[mask], y[mask]
        # Return filtered data
        return x, fyhat, fy

    def apply_condition(
        self, mask, cond, yhat, y=None, fnames=None, F=None, name=None
    ):
        """
        Apply the given condition to update the input mask (in place).

        :param mask: Boolean mask to be updated. It must specify `True` for
            points that must be considered for the evaluation, `False`
            otherwise.
        :type mask: :class:`np.ndarray` of bool
        :param cond: The condition specification.
        :type cond: dict
        :param yhat: The predictions.
        :type yhat: :class:`np.ndarray`
        :param y: The expected/reference classes.
        :type y: :class:`np.ndarray`
        :param fnames: The names of the features in the feature space matrix
            :math:`\pmb{F}`.
        :type fnames: list of str
        :param F: The feature space matrix.
        :type F: :class:`np.ndarray`
        :param name: The name of the filter to which the condition belongs.
        :type name: str
        """
        # Extract condition definition
        vname = cond.get('value_name', None)
        ctype = cond.get('condition_type', None)
        target = cond.get('value_target', None)
        preserve = cond.get('action', 'preserve').lower() == 'preserve'
        # Validation condition
        if vname is None:
            raise EvaluatorException(
                'AdvancedClassificationEvaluator failed to apply condition '
                f'from filter "{name}" because no value_name was given.'
            )
        if vname != "classification" and vname not in fnames:
            raise EvaluatorException(
                'AdvancedClassificationEvaluator failed to apply condition '
                f'from filter "{name}" because an unexpected value_name '
                f'("{vname}") was given.'
            )
        if ctype is None:
            raise EvaluatorException(
                'AdvancedClassificationEvaluator failed to apply condition '
                f'from filter "{name}" because no condition_type was given.'
            )
        if ctype is None:
            raise EvaluatorException(
                'AdvancedClassificationEvaluator failed to apply condition '
                f'from filter "{name}" because no value_target was given.'
            )
        # Extract condition value
        if vname == 'classification':
            fval = y
        elif vname == 'Prediction':
            fval = yhat
        else:
            found = False
            for i, fnamei in enumerate(fnames):
                if fnamei == vname:
                    fval = F[:, i]
                    found = True
                    break
            if not found:  # MUST NOT be reached (fnames already checked)
                raise EvaluatorException(
                    'AdvancedClassificationEvaluator failed to apply condition '
                    f'from filter "{name}" because an unexpected value_name '
                    f'("{vname}") was detected when extracting the '
                    'the condition\'s value.'
                )
        # Apply condition
        ctype_low = ctype.lower()
        cond_mask = None
        if ctype_low == 'not_equals':
            cond_mask = fval != target
        elif ctype_low == 'equals':
            cond_mask = fval == target
        elif ctype_low == 'less_than':
            cond_mask = fval < target
        elif ctype_low == 'less_than_or_equal_to':
            cond_mask = fval <= target
        elif ctype_low == 'greater_than':
            cond_mask = fval > target
        elif ctype_low == 'greater_than_or_equal_to':
            cond_mask = fval >= target
        elif ctype_low == 'in':
            cond_mask = np.array([fi in target for fi in fval], dtype=bool)
        elif ctype_low == 'not_in':
            cond_mask = np.array([fi not in target for fi in fval], dtype=bool)
        else:
            raise EvaluatorException(
                'AdvancedClassificationEvaluator failed to apply condition '
                f'from filter "{name}" because an unexpected condition_type'
                f'("{ctype}") was given.'
            )
        # Update filter mask (in place update)
        if preserve:  # on preserve
            mask[~cond_mask] = False
        else:  # on discard
            mask[cond_mask] = False

    def get_fnames_from_filters(self):
        """
        Obtain the names of the features that must be considered by the
        filters.

        :return: List with the names of the features that must be considered
            by the filters.
        :rtype: list of str
        """
        # Find unique feature names
        fnames = set()
        for f in self.filters:
            conditions = f.get('conditions', None)
            if conditions is None:
                continue
            for cond in conditions:
                vname = cond.get('value_name', None)
                if vname is not None:
                    fnames.add(vname)
        # Return
        return list(fnames)

    # ---  PIPELINE METHODS  --- #
    # -------------------------- #
    def eval_args_from_state(self, state):
        """
        Obtain the arguments to call the AdvancedClassificationEvaluator from
        the current pipeline's state.

        :param state: The pipeline's state.
        :type state: :class:`.SimplePipelineState`
        :return: The dictionary of arguments for calling
            AdvancedClassificationEvaluator.
        :rtype: dict
        """
        return {
            'pcloud': state.pcloud,
            'yhat': state.preds
        }
