# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.arch.architecture import Architecture
from src.model.deeplearn.arch.point_net import PointNet
from src.model.deeplearn.dlrun.hierarchical_pre_processor import \
    HierarchicalPreProcessor
from src.model.deeplearn.dlrun.hierarchical_post_processor import \
    HierarchicalPostProcessor
from src.model.deeplearn.layer.features_downsampling_layer import \
    FeaturesDownsamplingLayer
from src.model.deeplearn.layer.features_upsampling_layer import \
    FeaturesUpsamplingLayer
from src.model.deeplearn.layer.grouping_point_net_layer import \
    GroupingPointNetLayer
from src.model.deeplearn.layer.kpconv_layer import KPConvLayer
from src.model.deeplearn.layer.strided_kpconv_layer import StridedKPConvLayer
from src.model.deeplearn.layer.light_kpconv_layer import LightKPConvLayer
from src.model.deeplearn.layer.strided_light_kpconv_layer import \
    StridedLightKPConvLayer
from src.model.deeplearn.layer.hourglass_layer import HourglassLayer
from src.utils.dl_utils import DLUtils
from src.utils.dict_utils import DictUtils
from src.main.main_config import VL3DCFG
import tensorflow as tf
import numpy as np
import os


# ---   CLASS   --- #
# ----------------- #
class ConvAutoencPwiseClassif(Architecture):
    """
    :author: Alberto M. Esmoris Pena

    The convolutional autoencoder architecture for point-wise classification.

    Examples of convolutional autoencoders are the PointNet++ model
    (https://arxiv.org/abs/1706.02413) and the KPConv model
    (https://arxiv.org/abs/1904.08889).
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        See :meth:`architecture.Architecture.__init__`.
        """
        # Call parent's init
        if kwargs.get('arch_name', None) is None:
            kwargs['arch_name'] = 'ConvAutoenc_PointWise_Classification'
        super().__init__(**kwargs)
        # Set defaults from VL3DCFG
        kwargs = DictUtils.add_defaults(
            kwargs,
            VL3DCFG['MODEL']['ConvAutoencPwiseClassif']
        )
        # Assign attributes
        self.fnames = kwargs.get('fnames', None)
        if self.fnames is None:
            self.fnames = ['ones']  # If no features are given, use ones
        self.num_classes = kwargs.get('num_classes', None)
        pre_kwargs = kwargs.get('pre_processing')
        pre_kwargs['num_classes'] = self.num_classes
        self.pre_runnable = HierarchicalPreProcessor(**pre_kwargs)
        self.post_runnable = HierarchicalPostProcessor(self.pre_runnable)
        self.feature_extraction = kwargs.get('feature_extraction', None)
        pre_processor = self.pre_runnable.pre_processor
        self.structure_alignment = kwargs.get('structure_alignment', None)
        self.features_alignment = kwargs.get('features_alignment', None)
        self.num_downsampling_neighbors = \
            pre_processor.num_downsampling_neighbors
        self.num_pwise_neighbors = \
            pre_processor.num_pwise_neighbors
        self.num_upsampling_neighbors = \
            pre_processor.num_upsampling_neighbors
        self.downsampling_filter = kwargs.get(
            'downsampling_filter', 'mean'
        )
        self.upsampling_filter = kwargs.get(
            'upsampling_filter', 'mean'
        )
        self.upsampling_bn = kwargs.get('upsampling_bn', True)
        self.upsampling_bn_momentum = kwargs.get(
            'upsampling_bn_momentum', 0.0
        )
        self.upsampling_hourglass = kwargs.get('upsampling_hourglass', None)
        self.conv1d = kwargs.get('conv1d', True)
        self.conv1d_kernel_initializer = kwargs.get(
            'conv1d_kernel_initializer', 'glorot_normal'
        )
        self.output_kernel_initializer = kwargs.get(
            'output_kernel_initializer', 'glorot_normal'
        )
        self.max_depth = len(self.num_downsampling_neighbors)
        self.binary_crossentropy = False
        comp_args = kwargs.get('compilation_args', None)
        self.binary_crossentropy = DLUtils.is_using_binary_crossentropy(
            comp_args, default=False
        )
        # Cache-like attributes
        self.Xs, self.aligned_Xs = None, None
        self.F, self.aligned_F = None, None
        self.NDs, self.Ns, self.NUs = [None]*3
        self.skip_links = None
        self.last_downsampling_tensor = None
        self.last_upsampling_tensor = None
        self.kpconv_layers = None
        self.skpconv_layers = None
        self.lkpconv_layers = None
        self.slkpconv_layers = None
        self.parallel_hourglass_layers = None

    # ---   ARCHITECTURE METHODS   --- #
    # -------------------------------- #
    def build_input(self):
        r"""
        Build the input layer of the neural network. A convolutional
        autoencoder expects to receive many input tensors representing the
        hierarchical nature of the architecture. More concretely, for each
        element in the batch there must be:

        1)  The structure space matrices representing the points in the
            hierarchy of FPS receptive fields (typically, :math:`n_x=3`,
            i.e., 3D point clouds).

        .. math::
            \pmb{X}_1 \in \mathbb{R}^{R_1 \times n_x}, \ldots,
            \pmb{X}_{d^*} \in \mathbb{R}^{R_{d^*} \times n_x}

        2)  The feature space matrix representing the points in the first
            receptive field of the hierarchy.

        .. math::
            \pmb{F}_1 \in \mathbb{R}^{R_1 \times n_f}

        3)  The downsampling matrices after the first one (which is not used by
            the neural network itself but immediately before to transform the
            original input to the first receptive field).

        .. math::
            \pmb{N}^D_2 \in \mathbb{Z}^{R_2 \times K^D_2}, \ldots,
            \pmb{N}^D_{d^*} \in \mathbb{Z}^{R_{d^*} \times K^D_{d^*}}

        4)  The point-wise neighborhood matrices to be used at each downsampled
            representation as topological information.

        .. math::
            \pmb{N}_2 \in \mathbb{Z}^{R_2 \times K_2}, \ldots,
            \pmb{N}_{d^*} \in \mathbb{Z}^{R_{d^*} \times K_{d^*}}

        3)  The upsampling matrices after the first one (which is not used by
            the neural network itself but immediately after to transform the
            output from the first receptive field to the original space).

        .. math::
            \pmb{N}^U_2 \in \mathbb{Z}^{R_2 \times K^U_2}, \ldots,
            \pmb{N}^U_{d^*} \in \mathbb{Z}^{R_{d^*} \times K^U_{d^*}}

        :return: Built layers.
        :rtype: list of :class:`tf.Tensor`
        """
        # Handle coordinates as input (i.e., structure spaces)
        self.Xs = [
            tf.keras.layers.Input(shape=(None, 3), name=f'X_{d+1}')
            for d in range(self.max_depth)
        ]
        # Handle first receptive field input features
        self.F = tf.keras.layers.Input(
            shape=(None, len(self.fnames)),
            name='Fin'
        )
        # Handle downsampling matrices
        self.NDs = [
            tf.keras.layers.Input(
                shape=(None, self.num_downsampling_neighbors[d]),
                dtype='int32',
                name=f'ND_{d+1}'
            )
            for d in range(1, self.max_depth)
        ]
        # Handle point-wise neighborhood matrices
        self.Ns = [
            tf.keras.layers.Input(
                shape=(None, self.num_pwise_neighbors[d]),
                dtype='int32',
                name=f'N_{d+1}'
            )
            for d in range(self.max_depth)
        ]
        # Handle upsampling matrices
        self.NUs = [
            tf.keras.layers.Input(
                shape=(None, self.num_upsampling_neighbors[d]),
                dtype='int32',
                name=f'NU_{d+1}'
            )
            for d in range(1, self.max_depth)
        ]
        # Handle structure alignment
        if self.structure_alignment is not None:
            self.aligned_Xs = []
            for i, X in enumerate(self.Xs):
                self.aligned_Xs.append(PointNet.build_transformation_block(
                    X,
                    num_features=X.shape[-1],
                    name=f'X{i+1}_align',
                    tnet_pre_filters=self.structure_alignment[
                        'tnet_pre_filters_spec'
                    ],
                    tnet_post_filters=self.structure_alignment[
                        'tnet_post_filters_spec'
                    ],
                    kernel_initializer=self.structure_alignment.get(
                        'kernel_initializer', 'glorot_normal'
                    )
                ))
        # Handle features alignment
        if self.features_alignment is not None:
            self.aligned_F = PointNet.build_transformation_block(
                self.F,
                num_features=self.F.shape[-1],
                name='F_align',
                tnet_pre_filters=self.features_alignment[
                    'tnet_pre_filters_spec'
                ],
                tnet_post_filters=self.features_alignment[
                    'tnet_post_filters_spec'
                ],
                kernel_initializer=self.features_alignment.get(
                    'kernel_initializer', 'glorot_normal'
                )
            )
        # Return list of inputs
        # TODO Rethink : Do Xs[1:], NDs, Ns, and NUs need to be unrolled?
        return [self.Xs[0], self.F, self.Xs[1:], self.NDs, self.Ns, self.NUs]

    def build_hidden(self, x, **kwargs):
        """
        Build the hidden layers of the convolutional autoencoder neural
        network.

        :param x: The input layer for the first hidden layer.
        :type x: :class:`tf.Tensor`
        :return: The last hidden layer.
        :rtype: :class.`tf.Tensor`
        """
        # Downsampling hierarchy
        self.build_downsampling_hierarchy()
        # Upsampling hierarchy
        self.build_upsampling_hierarchy()
        # Return last hidden layer
        return self.last_upsampling_tensor

    def build_output(self, x, **kwargs):
        """
        Build the output layer of the convolutional autoencoder neural network
        for point-wise classification tasks.

        See :meth:`architecture.Architecture.build_output`.
        """
        # Handle output layer for binary cross-entropy loss
        if self.binary_crossentropy:
            return tf.keras.layers.Conv1D(
                1,
                kernel_size=1,
                activation='sigmoid',
                kernel_initializer=self.output_kernel_initializer,
                name='pwise_out'
            )(x)
        # Handle output layer for the general case
        return tf.keras.layers.Conv1D(
            self.num_classes,
            kernel_size=1,
            activation='softmax',
            kernel_initializer=self.output_kernel_initializer,
            name='pwise_out'
        )(x)

    # ---  CONVOLUTIONAL AUTOENCODER PWISE CLASSIF METHODS  --- #
    # --------------------------------------------------------- #
    def build_downsampling_hierarchy(self):
        """
        Build the downsampling hierarchy.

        :return: The last layer of the downsampling hierarchy.
        :rtype: :class:`tf.Tensor`
        """
        feat_extract_type = self.feature_extraction['type']
        feat_extract_type_low = feat_extract_type.lower()
        if feat_extract_type_low == 'pointnet':
            self.build_downsampling_pnet_hierarchy()
        elif feat_extract_type_low == 'kpconv':
            self.build_downsampling_kpconv_hierarchy()
        elif feat_extract_type_low == 'lightkpconv':
            self.build_downsampling_lightkpconv_hierarchy()
        else:
            raise DeepLearningException(
                f'ConvAutoencPwiseClassif received a "{feat_extract_type}" '
                'as type of feature extraction. It is not supported.'
            )

    def build_downsampling_pnet_hierarchy(self):
        """
        Build the downsampling hierarchy based on the PointNet operator.
        """
        self.skip_links = []
        i = 0
        ops_per_depth = self.feature_extraction['operations_per_depth']
        x = self.F if self.aligned_F is None else self.aligned_F
        Xs = self.Xs if self.aligned_Xs is None else self.aligned_Xs
        for _ in range(ops_per_depth[0]):
            x = GroupingPointNetLayer(
                self.feature_extraction['feature_space_dims'][i],
                H_activation=self.feature_extraction['H_activation'][i],
                H_initializer=self.feature_extraction['H_initializer'][i],
                H_regularizer=self.feature_extraction['H_regularizer'][i],
                H_constraint=self.feature_extraction['H_constraint'][i],
                gamma_activation=self.feature_extraction['gamma_activation'][i],
                gamma_kernel_initializer=self.feature_extraction[
                    'gamma_kernel_initializer'
                ][i],
                gamma_kernel_regularizer=self.feature_extraction[
                    'gamma_kernel_regularizer'
                ][i],
                gamma_kernel_constraint=self.feature_extraction[
                    'gamma_kernel_constraint'
                ][i],
                gamma_bias_enabled=self.feature_extraction[
                    'gamma_bias_enabled'
                ][i],
                gamma_bias_initializer=self.feature_extraction[
                    'gamma_bias_initializer'
                ][i],
                gamma_bias_regularizer=self.feature_extraction[
                    'gamma_bias_regularizer'
                ][i],
                gamma_bias_constraint=self.feature_extraction[
                    'gamma_bias_constraint'
                ][i],
                name=f'GPNet_d1_{i+1}'
            )([Xs[0], x, self.Ns[0]])
            if self.feature_extraction['bn']:
                x = tf.keras.layers.BatchNormalization(
                    momentum=self.feature_extraction['bn_momentum'],
                    name=f'GPNet_d1_{i+1}_BN'
                )(x)
            i += 1
        self.skip_links.append(x)
        for d in range(self.max_depth-1):
            x = FeaturesDownsamplingLayer(
                filter=self.downsampling_filter,
                name=f'DOWN_d{d+2}'
            )([
                Xs[d], Xs[d+1], x, self.NDs[d]
            ])
            for _ in range(ops_per_depth[d+1]):
                x = GroupingPointNetLayer(
                    self.feature_extraction['feature_space_dims'][i],
                    H_activation=self.feature_extraction['H_activation'][i],
                    H_initializer=self.feature_extraction['H_initializer'][i],
                    H_regularizer=self.feature_extraction['H_regularizer'][i],
                    H_constraint=self.feature_extraction['H_constraint'][i],
                    gamma_activation=self.feature_extraction['gamma_activation'][i],
                    gamma_kernel_initializer=self.feature_extraction[
                        'gamma_kernel_initializer'
                    ][i],
                    gamma_kernel_regularizer=self.feature_extraction[
                        'gamma_kernel_regularizer'
                    ][i],
                    gamma_kernel_constraint=self.feature_extraction[
                        'gamma_kernel_constraint'
                    ][i],
                    gamma_bias_enabled=self.feature_extraction[
                        'gamma_bias_enabled'
                    ][i],
                    gamma_bias_initializer=self.feature_extraction[
                        'gamma_bias_initializer'
                    ][i],
                    gamma_bias_regularizer=self.feature_extraction[
                        'gamma_bias_regularizer'
                    ][i],
                    gamma_bias_constraint=self.feature_extraction[
                        'gamma_bias_constraint'
                    ][i],
                    name=f'GPNet_d{d+2}_{i+1}'
                )([Xs[d+1], x, self.Ns[d+1]])
                if self.feature_extraction['bn']:
                    x = tf.keras.layers.BatchNormalization(
                        momentum=self.feature_extraction['bn_momentum'],
                        name=f'GPNet_d{d+2}_{i+1}_BN'
                    )(x)
                i += 1
            self.skip_links.append(x)
        self.last_downsampling_tensor = x

    def build_downsampling_kpconv_hierarchy(self):
        """
        Build the downsampling hierarchy based on the KPConv operator.
        """
        self.skip_links = []
        i = 0
        ops_per_depth = self.feature_extraction['operations_per_depth']
        x = self.F if self.aligned_F is None else self.aligned_F
        Xs = self.Xs if self.aligned_Xs is None else self.aligned_Xs
        self.kpconv_layers, self.skpconv_layers = [], []
        self.parallel_hourglass_layers = []
        for _ in range(ops_per_depth[0]):
            x, Dout = self.kpconv_prewrap(x, 1, i)
            kpcl = KPConvLayer(
                sigma=self.feature_extraction['sigma'][i],
                kernel_radius=self.feature_extraction['kernel_radius'][i],
                num_kernel_points=self.feature_extraction['num_kernel_points'][i],
                deformable=self.feature_extraction['deformable'][i],
                Dout=Dout,
                W_initializer=self.feature_extraction['W_initializer'][i],
                W_regularizer=self.feature_extraction['W_regularizer'][i],
                W_constraint=self.feature_extraction['W_constraint'][i],
                name=f'KPConv_d1_{i+1}'
            )
            self.kpconv_layers.append(kpcl)
            x = kpcl([Xs[0], x, self.Ns[0]])
            if self.feature_extraction['bn']:
                x = tf.keras.layers.BatchNormalization(
                    momentum=self.feature_extraction['bn_momentum'],
                    name=f'KPConv_d1_{i+1}_BN'
                )(x)
            if self.feature_extraction['activate']:
                x = tf.keras.layers.ReLU(
                    name=f'KPConv_d1_{i+1}_ReLU'
                )(x)
            x = self.kpconv_postwrap(x, 1, i)
            i += 1
        self.skip_links.append(x)
        for d in range(self.max_depth-1):
            x = self.build_kpconv_downsampling_layer(Xs, x, d, i)
            if self.feature_extraction['bn']:
                x = tf.keras.layers.BatchNormalization(
                    momentum=self.feature_extraction['bn_momentum'],
                    name=f'DOWN_d{d+2}_{i+1}_BN'
                )(x)
            if self.feature_extraction['activate']:
                x = tf.keras.layers.ReLU(
                    name=f'DOWN_d{d+2}_{i+1}_ReLU'
                )(x)
            for _ in range(ops_per_depth[d+1]):
                x, Dout = self.kpconv_prewrap(x, d+2, i)
                kpcl = KPConvLayer(
                    sigma=self.feature_extraction['sigma'][i],
                    kernel_radius=self.feature_extraction['kernel_radius'][i],
                    num_kernel_points=self.feature_extraction['num_kernel_points'][i],
                    deformable=self.feature_extraction['deformable'][i],
                    Dout=Dout,
                    W_initializer=self.feature_extraction['W_initializer'][i],
                    W_regularizer=self.feature_extraction['W_regularizer'][i],
                    W_constraint=self.feature_extraction['W_constraint'][i],
                    name=f'KPConv_d{d+2}_{i+1}'
                )
                self.kpconv_layers.append(kpcl)
                x = kpcl([Xs[d+1], x, self.Ns[d+1]])
                if self.feature_extraction['bn']:
                    x = tf.keras.layers.BatchNormalization(
                        momentum=self.feature_extraction['bn_momentum'],
                        name=f'KPConv_d{d+2}_{i+1}_BN'
                    )(x)
                if self.feature_extraction['activate']:
                    x = tf.keras.layers.ReLU(
                        name=f'KPConv_d{d+2}_{i+1}_ReLU'
                    )(x)
                x = self.kpconv_postwrap(x, d+2, i)
                i += 1
            self.skip_links.append(x)
        self.last_downsampling_tensor = x

    def build_kpconv_downsampling_layer(self, Xs, x, d, i):
        """
        Build a downsampling layer in the context of the KPConv and light
        KPConv models (i.e., support also :class:`.StridedKPConvLayer` and
        :class:`.StridedLightKPConvLayer` apart from
        :class:`.FeaturesDownsamplingLayer`).

        :param Xs: The list of receptive field-wise structure spaces.
        :type Xs: list
        :param x: The input features for the downsampling layer.
        :type x: :class:`tf.Tensor`
        :param d: The model depth.
        :type d: int
        :param i: The index of the operation (because there might be different
            number of operations per depth level).
        :type i: int
        :return: The downsampling layer.
        :rtype: :class:`tf.Tensor`
        """
        downsampling_filter = self.downsampling_filter.lower()
        if downsampling_filter == 'strided_kpconv':
            skpcl = StridedKPConvLayer(
                sigma=self.feature_extraction['sigma'][i],
                kernel_radius=self.feature_extraction['kernel_radius'][i],
                num_kernel_points=self.feature_extraction['num_kernel_points'][i],
                deformable=self.feature_extraction['deformable'][i],
                Dout=self.feature_extraction['feature_space_dims'][i],
                W_initializer=self.feature_extraction['W_initializer'][i],
                W_regularizer=self.feature_extraction['W_regularizer'][i],
                W_constraint=self.feature_extraction['W_constraint'][i],
                name=f'DOWN_SKPConv_d{d+2}_{i+1}'
            )
            self.skpconv_layers.append(skpcl)
            return skpcl([Xs[d], Xs[d+1], x, self.NDs[d]])
        elif downsampling_filter == 'strided_lightkpconv':
            slkpcl = StridedLightKPConvLayer(
                sigma=self.feature_extraction['sigma'][i],
                kernel_radius=self.feature_extraction['kernel_radius'][i],
                num_kernel_points=self.feature_extraction['num_kernel_points'][i],
                deformable=self.feature_extraction['deformable'][i],
                Dout=self.feature_extraction['feature_space_dims'][i],
                W_initializer=self.feature_extraction['W_initializer'][i],
                W_regularizer=self.feature_extraction['W_regularizer'][i],
                W_constraint=self.feature_extraction['W_constraint'][i],
                A_trainable=self.feature_extraction['A_trainable'][i],
                A_initializer=self.feature_extraction['A_initializer'][i],
                A_regularizer=self.feature_extraction['A_regularizer'][i],
                A_constraint=self.feature_extraction['A_constraint'][i],
                name=f'DOWN_SLKPConv_d{d + 2}_{i + 1}'
            )
            self.slkpconv_layers.append(slkpcl)
            return slkpcl([Xs[d], Xs[d + 1], x, self.NDs[d]])
        else:
            return FeaturesDownsamplingLayer(
                filter=self.downsampling_filter,
                name=f'DOWN_d{d+2}'
            )([Xs[d], Xs[d+1], x, self.NDs[d]])

    def build_downsampling_lightkpconv_hierarchy(self):
        """
        Build the downsampling hierarchy based on the light KPConv operator.
        """
        self.skip_links = []
        i = 0
        ops_per_depth = self.feature_extraction['operations_per_depth']
        x = self.F if self.aligned_F is None else self.aligned_F
        Xs = self.Xs if self.aligned_Xs is None else self.aligned_Xs
        self.lkpconv_layers, self.slkpconv_layers = [], []
        self.parallel_hourglass_layers = []
        for _ in range(ops_per_depth[0]):
            x, Dout = self.kpconv_prewrap(x, 1, i)
            lkpcl = LightKPConvLayer(
                sigma=self.feature_extraction['sigma'][i],
                kernel_radius=self.feature_extraction['kernel_radius'][i],
                num_kernel_points=self.feature_extraction['num_kernel_points'][i],
                deformable=self.feature_extraction['deformable'][i],
                Dout=Dout,
                W_initializer=self.feature_extraction['W_initializer'][i],
                W_regularizer=self.feature_extraction['W_regularizer'][i],
                W_constraint=self.feature_extraction['W_constraint'][i],
                A_trainable=self.feature_extraction['A_trainable'][i],
                A_initializer=self.feature_extraction['A_initializer'][i],
                A_regularizer=self.feature_extraction['A_regularizer'][i],
                A_constraint=self.feature_extraction['A_constraint'][i],
                name=f'LightKPConv_d1_{i+1}'
            )
            self.lkpconv_layers.append(lkpcl)
            x = lkpcl([Xs[0], x, self.Ns[0]])
            if self.feature_extraction['bn']:
                x = tf.keras.layers.BatchNormalization(
                    momentum=self.feature_extraction['bn_momentum'],
                    name=f'LightKPConv_d1_{i+1}_BN'
                )(x)
            if self.feature_extraction['activate']:
                x = tf.keras.layers.ReLU(
                    name=f'LightKPConv_d1_{i+1}_ReLU'
                )(x)
            x = self.kpconv_postwrap(x, 1, i)
            i += 1
        self.skip_links.append(x)
        for d in range(self.max_depth-1):
            x = self.build_kpconv_downsampling_layer(Xs, x, d, i)
            if self.feature_extraction['bn']:
                x = tf.keras.layers.BatchNormalization(
                    momentum=self.feature_extraction['bn_momentum'],
                    name=f'DOWN_d{d+2}_{i+1}_BN'
                )(x)
            if self.feature_extraction['activate']:
                x = tf.keras.layers.ReLU(
                    name=f'DOWN_d{d+2}_{i+1}_ReLU'
                )(x)
            for _ in range(ops_per_depth[d+1]):
                x, Dout = self.kpconv_prewrap(x, d+2, i)
                lkpcl = LightKPConvLayer(
                    sigma=self.feature_extraction['sigma'][i],
                    kernel_radius=self.feature_extraction['kernel_radius'][i],
                    num_kernel_points=self.feature_extraction['num_kernel_points'][i],
                    deformable=self.feature_extraction['deformable'][i],
                    Dout=Dout,
                    W_initializer=self.feature_extraction['W_initializer'][i],
                    W_regularizer=self.feature_extraction['W_regularizer'][i],
                    W_constraint=self.feature_extraction['W_constraint'][i],
                    A_trainable=self.feature_extraction['A_trainable'][i],
                    A_initializer=self.feature_extraction['A_initializer'][i],
                    A_regularizer=self.feature_extraction['A_regularizer'][i],
                    A_constraint=self.feature_extraction['A_constraint'][i],
                    name=f'LightKPConv_d{d+2}_{i+1}'
                )
                self.lkpconv_layers.append(lkpcl)
                x = lkpcl([Xs[d+1], x, self.Ns[d+1]])
                if self.feature_extraction['bn']:
                    x = tf.keras.layers.BatchNormalization(
                        momentum=self.feature_extraction['bn_momentum'],
                        name=f'LightKPConv_d{d+2}_{i+1}_BN'
                    )(x)
                if self.feature_extraction['activate']:
                    x = tf.keras.layers.ReLU(
                        name=f'LightKPConv_d{d+2}_{i+1}_ReLU'
                    )(x)
                x = self.kpconv_postwrap(x, d+2, i)
                i += 1
            self.skip_links.append(x)
        self.last_downsampling_tensor = x

    def build_upsampling_hierarchy(self):
        """
        Build the upsampling hierarchy.

        :return: The last layer of the upsampling hierarchy.
        :rtype: :class:`tf.Tensor`
        """
        x = self.last_downsampling_tensor
        Xs = self.Xs if self.aligned_Xs is None else self.aligned_Xs
        for d in range(self.max_depth-1):
            reverse_d = self.max_depth-2-d
            skip_link = self.skip_links[reverse_d]
            # Upsampling layer itself
            x = FeaturesUpsamplingLayer(
                filter=self.upsampling_filter,
                name=f'UP_d{reverse_d+2}'
            )([Xs[reverse_d+1], Xs[reverse_d], x, self.NUs[reverse_d]])
            x = tf.keras.layers.Concatenate(
                name=f'CONCAT_d{reverse_d+1}'
            )([x, skip_link])
            # 1D convolutions after upsampling
            filters = self.feature_extraction['feature_space_dims'][reverse_d]
            if self.conv1d:
                x = tf.keras.layers.Conv1D(
                    filters,
                    kernel_size=1,
                    strides=1,
                    padding="valid",
                    kernel_initializer=self.conv1d_kernel_initializer,
                    name=f'UpConv1D_d{reverse_d+1}'
                )(x)
            # Hourglass after upsampling
            if self.upsampling_hourglass is not None:
                Din = x.shape[-1]
                subspace_factor = self.upsampling_hourglass.get(
                    'subspace_factor', None
                )
                if subspace_factor is None:
                    raise DeepLearningException(
                        'ConvAutoencPwiseClassi only supports upsampling '
                        'hourglasses specified with subspace factor.'
                    )
                Dh = int(subspace_factor * max(Din, filters))
                x = HourglassLayer(
                    Dh,
                    filters,
                    activation=self.upsampling_hourglass['activation'],
                    activation2=self.upsampling_hourglass['activation2'],
                    regularize=self.upsampling_hourglass['regularize'],
                    spectral_strategy=self.upsampling_hourglass.get(
                        'spectral_strategy', 'approx'
                    ),
                    beta=self.upsampling_hourglass['loss_factor'],
                    W1_initializer=self.upsampling_hourglass['W1_initializer'],
                    W1_regularizer=self.upsampling_hourglass['W1_regularizer'],
                    W1_constraint=self.upsampling_hourglass['W1_constraint'],
                    W2_initializer=self.upsampling_hourglass['W2_initializer'],
                    W2_regularizer=self.upsampling_hourglass['W2_regularizer'],
                    W2_constraint=self.upsampling_hourglass['W2_constraint'],
                    name=f'UpHG_{reverse_d+1}'
                )(x)
            if self.upsampling_bn:
                x = tf.keras.layers.BatchNormalization(
                    momentum=self.upsampling_bn_momentum,
                    name=f'UpBN_d{reverse_d+1}'
                )(x)
            x = tf.keras.layers.Activation(
                "relu",
                name=f'UpReLU_d{reverse_d+1}'
            )(x)
        self.last_upsampling_tensor = x

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def __getstate__(self):
        """
        Method to be called when saving the serialized ConvAutoencPwiseClassif
        architecture.

        :return: The state's dictionary of the object.
        :rtype: dict
        """
        # Call parent's method
        state = super().__getstate__()
        # Add ConvAutoencPwiseClassif's attributes to state dictionary
        state['fnames'] = self.fnames
        state['num_classes'] = self.num_classes
        state['feature_extraction'] = self.feature_extraction
        state['structure_alignment'] = self.structure_alignment
        state['features_alignment'] = self.features_alignment
        state['downsampling_filter'] = self.downsampling_filter
        state['upsampling_filter'] = self.upsampling_filter
        state['upsampling_bn'] = self.upsampling_bn
        state['upsampling_bn_momentum'] = self.upsampling_bn_momentum
        state['conv1d_kernel_initializer'] = self.conv1d_kernel_initializer
        state['output_kernel_initializer'] = self.output_kernel_initializer
        state['max_depth'] = self.max_depth
        # Return
        return state

    def __setstate__(self, state):
        """
        Method to be called when loading and deserializing a previously
        serialized ConvAutoencPwiseClassif architecture.

        :param state: The state's dictionary of the saved
            ConvAutoencPwiseClassif architecture.
        :type state: dict
        :return: Nothing, but modifies the internal state of the object.
        """
        # Assign ConvAutoencPwiseClassif's attributes from state dictionary
        self.fnames = state['fnames']
        self.num_classes = state['num_classes']
        self.feature_extraction = state['feature_extraction']
        self.structure_alignment = state['structure_alignment']
        self.features_alignment = state['features_alignment']
        self.downsampling_filter = state['downsampling_filter']
        self.upsampling_filter = state['upsampling_filter']
        self.upsampling_bn = state['upsampling_bn']
        self.upsampling_bn_momentum = state['upsampling_bn_momentum']
        self.conv1d_kernel_initializer = state['conv1d_kernel_initializer']
        self.output_kernel_initializer = state['output_kernel_initializer']
        self.max_depth = state['max_depth']
        # Call parent's set state
        super().__setstate__(state)
        # Track KPConv layers
        self.kpconv_layers = [
            layer
            for layer in self.nn.layers
            if type(layer) == KPConvLayer
        ]
        # Track SKPConv layers
        self.skpconv_layers = [
            layer
            for layer in self.nn.layers
            if type(layer) == StridedKPConvLayer
        ]
        # Track light KPConv layers
        self.lkpconv_layers = [
            layer for layer in self.nn.layers
            if type(layer) == LightKPConvLayer
        ]
        self.slkpconv_layers = [
            layer
            for layer in self.nn.layers
            if type(layer) == StridedLightKPConvLayer
        ]

    # ---  KPCONV UTIL METHODS  --- #
    # ----------------------------- #
    def kpconv_prewrap(self, x, depth, idx):
        """
        Wrap the input before a KPConv layer with unary convolutions (also
        known as shared MLPs).

        :param x: The input to be wrapped.
        :param depth: The depth of the KPConv being pre-wrapped.
        :param idx: The index of the KPConv being pre-wrapped.
        :return: A tuple with the input for the next layer and the output
            dimensionality.
        :rtype: tuple
        """
        # Get KPConv wrapper specification
        wrap_spec = self.feature_extraction.get(
            'unary_convolution_wrapper', None
        )
        # Get Hourglass specification
        hourglass_spec = self.feature_extraction.get(
            'hourglass_wrapper', None
        )
        # Get input dimensionality
        Din = self.feature_extraction['feature_space_dims'][idx]
        # Default output with no wrappers at all
        out_x = x
        out_dim = self.feature_extraction['feature_space_dims'][idx]
        # Handle unary wrapper
        if wrap_spec is not None:
            out_x, out_dim = self.unary_convolution_prewrap(
                wrap_spec, Din, out_x, depth, idx
            )
            Din = out_dim  # Further wrappers need the updated input dim
        # Handle hourglass wrapper
        if hourglass_spec is not None:
            out_x, out_dim = self.hourglass_prewrap(
                hourglass_spec, Din, out_x, depth, idx
            )
            Din = out_dim  # Further wrappers need the updated input dim
        # Return
        return out_x, out_dim


    def kpconv_postwrap(self, x, depth, idx):
        """
        Wrap the output of a KPConv block with unary convolutions (also known
        as shared MLPs).

        :param x: The input to be wrapped.
        :param depth: The depth of the KPConv being post-wrapped.
        :param idx: The index of the KPConv being post-wrapped.
        :return: The input for the next layer
        """
        # Get KPConv wrapper specification
        wrap_spec = self.feature_extraction.get(
            'unary_convolution_wrapper', None
        )
        # Get Hourglass specification
        hourglass_spec = self.feature_extraction.get(
            'hourglass_wrapper', None
        )
        # Get output dimensionality
        Dout = self.feature_extraction['feature_space_dims'][idx]
        # Default output with no wrappers at all
        out_x = x
        # Handle unary wrapper
        if wrap_spec is not None:
            x = self.unary_convolution_postwrap(wrap_spec, Dout, x, depth, idx)
        # Handle hourglass wrapper
        if hourglass_spec is not None:
            x = self.hourglass_postwrap(hourglass_spec, Dout, x, depth, idx)
        # Return
        return x


    # ---  WRAPPER BLOCKS  --- #
    # ------------------------ #
    def unary_convolution_prewrap(self, wrap_spec, Din, x, depth, idx):
        """
        See :meth:`.ConvAutoencPwiseClassif.kpconv_prewrap`.
        """
        # Extract variables
        Dout = Din//wrap_spec.get('feature_dim_divisor', 2)
        # Build layers
        x = tf.keras.layers.Conv1D(
            Dout,
            kernel_size=1,
            padding="valid",
            kernel_initializer=wrap_spec.get('initializer', "glorot_uniform"),
            name=f'PreWrap_d{depth}_{idx+1}'
        )(x)
        if wrap_spec.get('bn', False):
            x = tf.keras.layers.BatchNormalization(
                momentum=wrap_spec.get('bn_momentum', 0.0),
                name=f'PreWrap_d{depth}_{idx+1}_BN'
            )(x)
        x = tf.keras.layers.Activation(
            wrap_spec.get('activation', 'relu'),
            name=f'PreWrap_d{depth}_{idx+1}_ACT'
        )(x)
        # Return
        return x, Dout

    def unary_convolution_postwrap(self, wrap_spec, Dout, x, depth, idx):
        """
        See :meth:`.ConvAutoencPwiseClassif.unary_convolution_postwrap`.
        """
        # Build layers
        x = tf.keras.layers.Conv1D(
            Dout,
            kernel_size=1,
            padding="valid",
            kernel_initializer=wrap_spec.get('initializer', "glorot_uniform"),
            name=f'PostWrap_d{depth}_{idx+1}'
        )(x)
        if wrap_spec.get('bn', False):
            x = tf.keras.layers.BatchNormalization(
                momentum=wrap_spec.get('bn_momentum', 0.0),
                name=f'PostWrap_d{depth}_{idx+1}_BN'
            )(x)
        x = tf.keras.layers.Activation(
            wrap_spec.get('activation', 'relu'),
            name=f'PostWrap_d{depth}_{idx+1}_ACT'
        )(x)
        # Return
        return x


    def hourglass_prewrap(self, hourglass_spec, Din, x, depth, idx):
        """
        See :meth:`.ConvAutoencPwiseClassif.kpconv_prewrap`.
        """
        # Extract variables
        subspace_factor = hourglass_spec.get('subspace_factor', None)
        dim_div = hourglass_spec.get('feature_dim_divisor', 4)
        Dout = Din//dim_div
        Dout_par = Din
        if subspace_factor is not None:
            Dh = int(subspace_factor * max(Din, Dout))
            Dh_par = int(subspace_factor * max(Din, Dout_par))
        else:
            Dh = int(hourglass_spec['internal_dim'][idx])
            Dh_par = int(hourglass_spec['parallel_internal_dim'][idx])
        # Build parallel hourglass
        x_par = HourglassLayer(
            Dh_par,
            Dout_par,
            activation=hourglass_spec['activation'][idx],
            activation2=hourglass_spec['activation2'][idx],
            regularize=hourglass_spec['regularize'][idx],
            spectral_strategy=hourglass_spec.get(
                'spectral_strategy', 'approx'
            ),
            beta=hourglass_spec['loss_factor'],
            W1_initializer=hourglass_spec['W1_initializer'][idx],
            W1_regularizer=hourglass_spec['W1_regularizer'][idx],
            W1_constraint=hourglass_spec['W1_constraint'][idx],
            W2_initializer=hourglass_spec['W2_initializer'][idx],
            W2_regularizer=hourglass_spec['W1_regularizer'][idx],
            W2_constraint=hourglass_spec['W1_constraint'][idx],
            name=f'ParHG_d{depth}_{idx+1}'
        )(x)
        # Build hourglass
        x = HourglassLayer(
            Dh,
            Dout,
            activation=hourglass_spec['activation'][idx],
            activation2=hourglass_spec['activation2'][idx],
            regularize=hourglass_spec['regularize'][idx],
            spectral_strategy=hourglass_spec.get(
                'spectral_strategy', 'approx'
            ),
            beta=hourglass_spec['loss_factor'],
            W1_initializer=hourglass_spec['W1_initializer'][idx],
            W1_regularizer=hourglass_spec['W1_regularizer'][idx],
            W1_constraint=hourglass_spec['W1_constraint'][idx],
            W2_initializer=hourglass_spec['W2_initializer'][idx],
            W2_regularizer=hourglass_spec['W1_regularizer'][idx],
            W2_constraint=hourglass_spec['W1_constraint'][idx],
            name=f'PreHG_d{depth}_{idx+1}'
        )(x)
        # Handle batch normalization
        if hourglass_spec.get('bn', False):
            # When applying batch normalization, it is recommended to build
            # the hourglass with sigma2 (second activation) as the identity.
            x = tf.keras.layers.BatchNormalization(
                momentum=hourglass_spec.get('bn_momentum', 0.0),
                name=f'PreHG_d{depth}_{idx+1}_BN'
            )(x)
            x_par = tf.keras.layers.BatchNormalization(
                momentum=hourglass_spec.get('bn_momentum', 0.0),
                name=f'ParHG_d{depth}_{idx + 1}_BN'
            )(x_par)
            act2 = hourglass_spec['activation2'][idx]
            if act2 is not None and act2.lower() != 'identity':
                x = tf.keras.layers.Activation(
                    act2,
                    name=f'PreHG_d{depth}_{idx+1}_ACT'
                )(x)
                x_par = tf.keras.layers.Activation(
                    act2,
                    name=f'ParHG_d{depth}_{idx + 1}_ACT'
                )(x_par)
        # Register parallel hourglass so it can be handled during postwrap
        self.parallel_hourglass_layers.append(x_par)
        # Return
        return x, Dout

    def hourglass_postwrap(self, hourglass_spec, Dout, x, depth, idx):
        """
        See :meth:`.ConvAutoencPwiseClassif.kpconv_postwrap`.
        """
        # Extract variables
        Din = x.shape[-1]
        subspace_factor = hourglass_spec.get('subspace_factor', None)
        if subspace_factor is not None:
            Dh = int(subspace_factor * max(Din, Dout))
        else:
            Dh = int(hourglass_spec['internal_dim'][idx])
        # Build hourglass
        x = HourglassLayer(
            Dh,
            Dout,
            activation=hourglass_spec['activation'][idx],
            activation2=hourglass_spec['activation2'][idx],
            regularize=hourglass_spec['regularize'][idx],
            spectral_strategy=hourglass_spec.get(
                'spectral_strategy', 'approx'
            ),
            beta=hourglass_spec['loss_factor'],
            W1_initializer = hourglass_spec['W1_initializer'][idx],
            W1_regularizer = hourglass_spec['W1_regularizer'][idx],
            W1_constraint = hourglass_spec['W1_constraint'][idx],
            W2_initializer = hourglass_spec['W2_initializer'][idx],
            W2_regularizer = hourglass_spec['W1_regularizer'][idx],
            W2_constraint = hourglass_spec['W1_constraint'][idx],
            name = f'PostHG_d{depth}_{idx + 1}'
        )(x)
        # Linear superposition wrt parallel hourglass
        x_par = self.parallel_hourglass_layers[idx]
        x = tf.keras.layers.Add()([x, x_par])
        # Batch normalization
        if hourglass_spec.get('out_bn', True):
            x = tf.keras.layers.BatchNormalization(
                momentum=hourglass_spec.get('out_bn_momentum', 0.98),
                name=f'PostHG_d{depth}_{idx+1}_BN'
            )(x)
        # ReLU activation
        x = tf.keras.layers.Activation(
            hourglass_spec.get('activation', 'relu')[idx],
            name=f'PostHG_d{depth}_{idx+1}_ACT'
        )(x)
        # Return
        return x

    # ---  FIT LOGIC CALLBACKS  --- #
    # ----------------------------- #
    def prefit_logic_callback(self, cache_map):
        """
        The callback implementing any necessary logic immediately before
        fitting a ConvAutoencPwiseClassif model.

        :param cache_map: The key-word dictionary containing variables that
            are guaranteed to live at least during prefit, fit, and postfit.
        :return: Nothing.
        """
        # Prefit logic for KPConv layer representation
        if(
            self.kpconv_layers is not None and
            cache_map.get('kpconv_representation_dir', None) is not None
        ):
            cache_map['kpconv_Wpast'] = []
            for i, kpconv_layer in enumerate(self.kpconv_layers):
                kpconv_layer.export_representation(
                    os.path.join(
                        cache_map['kpconv_representation_dir'],
                        f'INIT_{kpconv_layer.name}'
                    ),
                    out_prefix=cache_map['out_prefix'],
                    Wpast=None
                )
                cache_map['kpconv_Wpast'].append(np.array(kpconv_layer.W))
        # Prefit logic for Strided KPConv layer representation
        if(
            self.skpconv_layers is not None and
            cache_map.get('skpconv_representation_dir', None) is not None
        ):
            cache_map['skpconv_Wpast'] = []
            for i, skpconv_layer in enumerate(self.skpconv_layers):
                skpconv_layer.export_representation(
                    os.path.join(
                        cache_map['skpconv_representation_dir'],
                        f'INIT_{skpconv_layer.name}'
                    ),
                    out_prefix=cache_map['out_prefix'],
                    Wpast=None
                )
                cache_map['skpconv_Wpast'].append(np.array(skpconv_layer.W))
        # Prefit logic for Light KPConv layer representation
        if(
            self.lkpconv_layers is not None and
            cache_map.get('lkpconv_representation_dir', None) is not None
        ):
            cache_map['lkpconv_Wpast'] = []
            cache_map['lkpconv_Apast'] = []
            for i, lkpconv_layer in enumerate(self.lkpconv_layers):
                lkpconv_layer.export_representation(
                    os.path.join(
                        cache_map['lkpconv_representation_dir'],
                        f'INIT_{lkpconv_layer.name}'
                    ),
                    out_prefix=cache_map['out_prefix'],
                    Wpast=None
                )
                cache_map['lkpconv_Wpast'].append(np.array(lkpconv_layer.W))
                cache_map['lkpconv_Apast'].append(np.array(lkpconv_layer.A))
        # Prefit logic for Strided Light KPConv layer representation
        if(
            self.slkpconv_layers is not None and
            cache_map.get('slkpconv_representation_dir', None) is not None
        ):
            cache_map['slkpconv_Wpast'] = []
            cache_map['slkpconv_Apast'] = []
            for i, slkpconv_layer in enumerate(self.slkpconv_layers):
                slkpconv_layer.export_representation(
                    os.path.join(
                        cache_map['slkpconv_representation_dir'],
                        f'INIT_{slkpconv_layer.name}'
                    ),
                    out_prefix=cache_map['out_prefix'],
                    Wpast=None
                )
                cache_map['slkpconv_Wpast'].append(np.array(slkpconv_layer.W))
                cache_map['slkpconv_Apast'].append(np.array(slkpconv_layer.A))

    def posfit_logic_callback(self, cache_map):
        """
        The callback implementing any necessary logic immediately after
        fitting a ConvAutoencPwiseClassif model.

        :param cache_map: The key-word dictionary containing variables that
            are guaranteed to live at least during prefit, fit, and postfit.
        :return: Nothing.
        """
        # Postfit logic for KPConv layer representation
        if(
            self.kpconv_layers is not None and
            cache_map.get('kpconv_representation_dir')
        ):
            for i, kpconv_layer in enumerate(self.kpconv_layers):
                kpconv_layer.export_representation(
                    os.path.join(
                        cache_map['kpconv_representation_dir'],
                        f'TRAINED_{kpconv_layer.name}'
                    ),
                    out_prefix=cache_map['out_prefix'],
                    Wpast=cache_map['kpconv_Wpast'][i]
                )
        # Postfit logic Strided KPConv layer representation
        if(
            self.skpconv_layers is not None and
            cache_map.get('skpconv_representation_dir', None) is not None
        ):
            for i, skpconv_layer in enumerate(self.skpconv_layers):
                skpconv_layer.export_representation(
                    os.path.join(
                        cache_map['skpconv_representation_dir'],
                        f'TRAINED_{skpconv_layer.name}'
                    ),
                    out_prefix=cache_map['out_prefix'],
                    Wpast=cache_map['skpconv_Wpast'][i]
                )
        # Postfit logic for Light KPConv layer representation
        if(
            self.lkpconv_layers is not None and
            cache_map.get('lkpconv_representation_dir')
        ):
            for i, lkpconv_layer in enumerate(self.lkpconv_layers):
                lkpconv_layer.export_representation(
                    os.path.join(
                        cache_map['lkpconv_representation_dir'],
                        f'TRAINED_{lkpconv_layer.name}'
                    ),
                    out_prefix=cache_map['out_prefix'],
                    Wpast=cache_map['lkpconv_Wpast'][i],
                    Apast=cache_map['lkpconv_Apast'][i]
                )
        # Postfit logic Strided Light KPConv layer representation
        if(
            self.slkpconv_layers is not None and
            cache_map.get('slkpconv_representation_dir', None) is not None
        ):
            for i, slkpconv_layer in enumerate(self.slkpconv_layers):
                slkpconv_layer.export_representation(
                    os.path.join(
                        cache_map['slkpconv_representation_dir'],
                        f'TRAINED_{slkpconv_layer.name}'
                    ),
                    out_prefix=cache_map['out_prefix'],
                    Wpast=cache_map['slkpconv_Wpast'][i],
                    Apast=cache_map['slkpconv_Apast'][i]
                )

