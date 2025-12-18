# ---   IMPORTS   --- #
# ------------------- #
from src.utils.ptransf.point_transformer import PointTransformer, \
    PointTransformerException
from src.utils.neighborhood.support_neighborhoods import SupportNeighborhoods
from src.utils.dict_utils import DictUtils
import src.main.main_logger as LOGGING
from scipy.spatial import KDTree as KDT
import numpy as np
import itertools
import time


# ---   CLASS   --- #
# ----------------- #
class PointCloudSampler(PointTransformer):
    """
    :author: Alberto M. Esmoris Pena

    Class for transforming points following one or many sampling strategies.

    :ivar neighborhood_sampling: The specification of the neighborhood sampling
        strategy, if any.

        The format for this attribute should be like:

        .. code-block:: json

            {
                "support_conditions": [
                    {
                        "value_name": "ClassAmbiguity",
                        "condition_type": "greater_than_or_equal_to",
                        "value_target": 0.5,
                        "action": "preserve"
                    }
                ],
                "support_strategy": "fps",
                "support_strategy_num_points": 100000,
                "support_strategy_fast": true,
                "support_chunk_size": 50000,
                "center_on_pcloud": false,
                "neighborhood": {
                    "type": "sphere",
                    "radius": 2.5,
                    "separation_factor": 0
                },
                "neighborhoods_per_iter": 10000,
                "nthreads": -1
            }

    :vartype neighborhood_sampling: dict or None
    """
    # ---  EXTRACT FROM SPEC  --- #
    # --------------------------- #
    @staticmethod
    def extract_ptransf_args(spec):
        """
        Extract the arguments to initialize/instantiate a PointCloudSampler.

        :param spec: The key-word specificaiton containing the arguments.
        :return: The arguments to initialize/instantiate a PointCloudSampler.
        :rtype: dict
        """
        # Initialize from parent
        kwargs = PointTransformer.extract_ptransf_args(spec)
        # Extract particular arguments of PointCloudSampler
        kwargs['neighborhood_sampling'] = spec.get(
            'neighborhood_sampling', None
        )
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a PointCloudSampler.

        :param kwargs: The attributes for the PointCloudSampler.
        """
        # Call parent init
        super().__init__(**kwargs)
        # Assign attributes
        self.neighborhood_sampling = kwargs.get('neighborhood_sampling', None)

    # ---   POINT TRANSFORM METHODS   --- #
    # ----------------------------------- #
    def transform(
        self, X=None, F=None, y=None, yhat=None, out_prefix=None, **kwargs
    ):
        """
        The fundamental point transform logic defining the point cloud sampler.

        See :class:`.PointTransformer` and :meth:`.PointTransformer.transform`.
        """
        # Start measuring time
        m_in = F.shape[0] if X is None else X.shape[0]  # Number of input points
        start = time.perf_counter()
        # Initialize list of selected indices
        selected_indices = [i for i in range(m_in)]
        # Apply the different sampling strategies
        if self.neighborhood_sampling:
            selected_indices, X, F, y, yhat = \
                PointCloudSampler.update_selected_indices(
                    selected_indices,
                    *self.apply_neighborhood_sampling(
                        X, F, y, yhat, out_prefix=out_prefix, **kwargs
                    )
                )
        # Report transformation time
        end = time.perf_counter()
        m_out = X.shape[0]
        LOGGING.LOGGER.info(
            f'PointCloudSampler transformed {m_in} points into {m_out} points '
            f'in {end-start:.3f} seconds.'
        )
        # Return
        return selected_indices

    def transform_pcloud(self, pcloud, out_prefix=None, **kwargs):
        """
        Update the input point cloud in place.

        See :meth:`.PointCloudSampler.transform` and
        :meth:`.PointTransformer.transform_pcloud`.
        """
        # Transform the point cloud to obtain selected indices
        selected_indices = super().transform_pcloud(
            pcloud, out_prefix=out_prefix, **kwargs
        )
        # Update the input point cloud in place
        return pcloud.preserve_mask(selected_indices)

    # ---   SAMPLING METHODS   --- #
    # ---------------------------- #
    def apply_neighborhood_sampling(
        self, X, F, y, yhat, out_prefix=None, **kwargs
    ):
        """
        Compute a neighborhood sampling. This strategy considers each support
        point (for example, the entire X, or a subset of X that satisfied the
        given conditions) and computes it neighborhood. The indices of the
        points in the neighborhoods are then returned.

        :param X: The structure space matrix representing the point cloud to be
            sampled.
        :type X: :class:`np.ndarray`
        :param F: the feature space matrix representing the point cloud to be
            sampled.
        :type F: :class:`np.ndarray` or None
        :param y: The references of the point cloud to be sampled (typically
            a vector of classes).
        :type y: :class:`np.ndarray` or None
        :param yhat: The predictions on the point cloud to be sampled (
            typically a vector of classes).
        :type yhat: :class:`np.ndarray` or None
        :param out_prefix: See :class:`.PointTransformer` and
            :meth:`.PointTransformer.transform`.
        :type kwargs: dict
        :return: List with the indices of the points to be included in the
            sampled version of the point cloud.
        :rtype: List of int
        """
        # Start time measurement
        start = time.perf_counter()
        # Validations before computations
        if 'support_conditions' not in self.neighborhood_sampling:
            raise PointTransformerException(
                'PointCloudSampler cannot apply neighborhood sampling without '
                'support_conditions specification.'
            )
        if 'neighborhood' not in self.neighborhood_sampling:
            raise PointTransformerException(
                'PointCloudSampler cannot apply neighborhood sampling without '
                'neighborhood specification.'
            )
        # Extract custom support points (if necessary)
        X_sup = None
        if (
            self.neighborhood_sampling['neighborhood'].get(
                'separation_factor', 0
            ) == 0
        ):
            X_sup = self.filter_support_points(
                self.neighborhood_sampling['support_conditions'],
                X, F, y, yhat,
                min_distance=self.neighborhood_sampling['support_min_distance'],
                out_prefix=out_prefix
            )
        # Precompute KDTree for structure space
        ntype = self.neighborhood_sampling['neighborhood']['type']
        ntype_low = ntype.lower()
        if ntype_low == 'cylinder' or ntype_low == 'rectangular2d':
            kdt = KDT(X[:, :2])
        elif ntype_low == 'sphere' or ntype_low == 'rectangular3d':
            kdt = KDT(X)
        else:
            raise PointTransformerException(
                'PointCloudSampler cannot apply neighborhood sampling '
                f'due to an unexpected neighborhood type: {ntype}'
            )
        # Compute neighborhoods
        I = set()
        num_neighborhoods = len(X_sup)
        neighborhoods_per_iter = self.neighborhood_sampling.get(
            'neighborhoods_per_iter', 0
        )
        num_iters = 1 if neighborhoods_per_iter == 0 else (
            int(np.ceil(num_neighborhoods / neighborhoods_per_iter))
        )
        if num_iters == 1:
            neighborhoods_per_iter = num_neighborhoods
        for iter in range(num_iters):
            # Extract start (a) and end (b) indices
            a = iter*neighborhoods_per_iter
            b = (iter+1)*neighborhoods_per_iter
            if X_sup is None:  # Use SupportNeighborhoods if no custom support
                Ii = SupportNeighborhoods(
                    self.neighborhood_sampling['neighborhood'],
                    support_strategy=self.neighborhood_sampling.get(
                        'support_strategy', 'grid'
                    ),
                    support_strategy_num_points=self.neighborhood_sampling.get(
                        'support_strategy_num_points', 1000
                    ),
                    support_strategy_fast=self.neighborhood_sampling.get(
                        'support_strategy_fast', False
                    ),
                    support_chunk_size=self.neighborhood_sampling.get(
                        'support_chunk_size', 0
                    ),
                    center_on_pcloud=self.neighborhood_sampling.get(
                        'center_on_pcloud', False
                    ),
                    nthreads=self.neighborhood_sampling.get(
                        'nthreads', 1
                    )
                ).compute(X, kdt=kdt)[1]
            else:  # Compute neighborhoods from custom support
                Ii = self.compute_neighborhoods(
                    X,
                    X_sup[a:b],
                    self.neighborhood_sampling['neighborhood'],
                    support_chunk_size=self.neighborhood_sampling.get(
                        'support_chunk_size', 0
                    ),
                    kdt=kdt
                )
            # Consider neighbors in the set of indices
            I.update(itertools.chain.from_iterable(Ii))
        # Report time
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'PointCloudSampler computed neighborhood sampling in '
            f'{end-start:.3f} seconds.'
        )
        # Return
        return list(I), X, F, y, yhat

    # ---   POINT CLOUD SAMPLER UTILS  --- #
    # ------------------------------------ #
    @staticmethod
    def update_selected_indices(
        base_selected_indices,
        new_selected_indices,
        X, F, y, yhat
    ):
        """
        Merge the current selected indices with the new selected indices.
        Note that the resulting selected indices are built by selecting those
        indices at ``base_selected_indices`` specified by
        ``new_selected_indices``.

        :return: The new list of selected indices.
        :rtype: list
        """
        # Generate new selected indices
        selected_indices = np.array(
            base_selected_indices
        )[new_selected_indices].tolist()
        # Update point cloud
        X = None if X is None else X[selected_indices]
        F = None if F is None else F[selected_indices]
        y = None if y is None else y[selected_indices]
        yhat = None if yhat is None else yhat[selected_indices]
        # Return updated selected indices and point cloud
        return selected_indices, X, F, y, yhat

    def filter_support_points(
        self, conditions, X, F, y, yhat, min_distance=None, out_prefix=None
    ):
        r"""
        Filter the given point cloud considering the given conditions
        to obtain the corresponding support points.

        :param conditions: A list with the dictionary-like specifications of
            each condition that must be applied. Each condition must specify:

            -- ``"value_name"`` : The name of the value involved in the
                condition.

            -- ``"condition_type"`` : The relational governing the condition,
                i.e., ``"not_equals"`` (:math:`\neq`),
                ``"equals"`` (:math:`=`),
                ``"less_than"`` (:math:`<`),
                ``"less_than_or_equal_to"`` (:math:`\leq`),
                ``"greater_than"`` (:math:`>`),
                ``"greater_than_or_equal_to"`` (:math:`\geq`),
                ``"in"`` (:math:`\in`), or
                ``"not_in"`` (:math:`\notin`).

            -- ``"value_target"`` : The value for the rhs of the relational.
                Note that the lhs will be the corresponding value from the
                point cloud.

            -- ``"action"`` : Whether the conditions define points that must be
                preserved (``"preserve"``) or discarded (``"discard"``).

        :type conditions: list of dict
        :param X: The structure space matrix representing the point cloud.
        :type X: :class:`np.ndarray` or None
        :param F: The feature space matrix representing the point cloud.
        :type F: :class:`np.ndarray` or None
        :param y: The references of the point cloud.
        :type y: :class:`np.ndarray` or None
        :param yhat: The predictions of the point cloud.
        :type yhat: :class:`np.ndarray` or None
        :param min_distance: When given, support points that are closer to
            other support points in less than min_distance will be discarded.
        :type min_distance: float or None
        :param out_prefix: See :class:`.PointTransformer` and
            :meth:`.PointTransformer.transform`.
        :return: The structure space matrix representing the support points.
        :rtype: :class:`np.ndarray`
        """
        # If no filtering is required, X already gives the support points
        if conditions is None or len(conditions) < 1:
            return X
        # Apply each condition
        sup_X, sup_F, sup_y, sup_yhat = X, F, y, yhat
        for cond in conditions:
            sup_X, sup_F, sup_y, sup_yhat = self.apply_condition(
                cond, sup_X, sup_F, sup_y, sup_yhat
            )
        # Apply distance filter, if requested
        if min_distance is not None and min_distance > 0:
            start = time.perf_counter()
            sup_m_before = sup_X.shape[0]
            # Build KDTree and prepare indices
            kdt = KDT(sup_X)
            k = 0
            I = np.array([  # Indices to be preserved
                i for i in range(sup_X.shape[0])
            ])
            # Check there is at least one support point
            if len(I) < 1:
                raise PointTransformerException(
                    'PointCloudSampler could not sample even a single support '
                    f'point. Therefore, no sampling is possible.'
                )
            # Filter iteratively by min distance
            while k < len(I):
                i = I[k]  # Get next point index
                xi = sup_X[i]  # The coordinates of the point
                Ii = kdt.query_ball_point(xi, min_distance)  # Neighborhood
                Ii.remove(i)  # Remove center point
                I = np.setdiff1d(I, Ii)
                k += 1
            sup_X = sup_X[I]
            # Log results
            sup_m_after = sup_X.shape[0]
            end = time.perf_counter()
            LOGGING.LOGGER.info(
                f'PointCloudSampler selected {sup_m_after} from '
                f'{sup_m_before} support points '
                f'(filtered out {sup_m_before-sup_m_after} points) '
                f'by min distance ({min_distance:.6f}) '
                f'in {end-start:.3f} seconds.'
            )
        # Return
        return sup_X

    def apply_condition(self, cond, X, F, y, yhat):
        """
        Apply the given condition to the given point cloud.
        See :meth:`.PointCloudSampler.filter_support_points` for further
        details when using this function to filter the support points.

        :return: The structure space, feature space, references, and
            predictions of the filtered point cloud.
        :rtype: tuple of :class:`np.ndarray`
        """
        # Find value for the condition
        fidx = np.flatnonzero(cond['value_name'] == self._fnames)
        if len(fidx) < 1:
            vname = cond['value_name']
            vname_low = vname.lower()
            if vname_low == 'classification':
                f = y
            elif vname_low == 'prediction':
                f = yhat
            else:
                raise PointTransformerException(
                    'PointCloudSampler failed to applied condition because '
                    f'an unexpected value_name was found: "{vname}"'
                )
        else:
            f = F[:, fidx]
        f = f.flatten()
        # Check condition
        target = cond['value_target']
        ctype = cond['condition_type']
        ctype_low = ctype.lower()
        if ctype_low == 'not_equals':
            mask = f != target
        elif ctype_low == 'equals':
            mask = f == target
        elif ctype_low == 'less_than':
            mask = f < target
        elif ctype_low == 'less_than_or_equal_to':
            mask = f <= target
        elif ctype_low == 'greater_than':
            mask = f > target
        elif ctype_low == 'greater_than_or_equal_to':
            mask = f >= target
        elif ctype_low == 'in':
            mask = f in target
        elif ctype_low == 'notin':
            mask = f not in target
        else:
            raise PointTransformerException(
                'PointCloudSampler failed to apply condition due to an '
                f'unexpected condition type: "{ctype}"'
            )
        # Apply condition
        if cond.get('action', 'preserve').lower() == 'discard':
            mask = ~mask
        if X is not None:
            X = X[mask]
        if F is not None:
            F = F[mask]
        if y is not None:
            y = y[mask]
        if yhat is not None:
            yhat = yhat[mask]
        # Return
        return X, F, y, yhat

    @staticmethod
    def compute_neighborhoods(
        X, X_sup, neighborhood_spec, support_chunk_size=0, kdt=None
    ):
        """
        Compute the neighborhood of each support point on X.

        See :class:`.SupportNeighborhoods` for more details.

        :param X_sup: The support structure space.
        :type X_sup: :class:`np.ndarray`
        :param X: The structure space where the neighbors must be.
        :type X: :class:`np.ndarray`
        :param neighborhood_spec: The key-word specification of the
            neighborhood.
        :type neighborhood_spec: dict
        :param support_chunk_size: See :class:`.SupportNeighborhoods`.
        :type support_chunk_size: int
        :param kdt: See :meth:`.SupportNeighborhoods.compute`.
        """
        # Handle neighborhood finding
        ngbhd_type = neighborhood_spec['type']
        ngbhd_type_low = ngbhd_type.lower()
        # Compute the neighborhood depending on the type
        if neighborhood_spec['radius'] == 0:
            # The neighborhood of radius 0 is said to be the entire point cloud
            I = [np.arange(len(X), dtype=int).tolist()]
        elif ngbhd_type_low == 'cylinder':
            I = SupportNeighborhoods.compute_cylindrical_neighborhoods(
                X[:, :2],
                X_sup[:, :2],
                neighborhood_spec,
                kdt=kdt
            )
        elif ngbhd_type_low == 'sphere':
            I = SupportNeighborhoods.compute_spherical_neighborhoods(
                X,
                X_sup,
                neighborhood_spec,
                kdt=kdt
            )
        elif ngbhd_type_low == 'rectangular2d':
            I = SupportNeighborhoods.compute_rectangular2D_neighborhoods(
                X[:, :2],
                X_sup[:, :2],
                neighborhood_spec,
                kdt=kdt
            )
        elif ngbhd_type_low == 'rectangular3d':
            I = SupportNeighborhoods.compute_rectangular3D_neighborhoods(
                X,
                X_sup,
                neighborhood_spec,
                support_chunk_size=support_chunk_size,
                kdt=kdt
            )
        else:
            raise ValueError(
                'PointCloudSampler object does not expect a '
                f'neighborhood specification of type "{ngbhd_type}"'
            )
        # Return found neighborhood
        return I
