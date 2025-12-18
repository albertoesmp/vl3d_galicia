# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.arch.architecture import Architecture
from src.model.deeplearn.dlrun.hierarchical_sg_pre_processorpp import \
    HierarchicalSGPreProcessorPP
from src.model.deeplearn.dlrun.hierarchical_sg_post_processorpp import \
    HierarchicalSGPostProcessorPP
from src.model.deeplearn.layer.submanifold_spconv3d_layer import \
    SubmanifoldSpConv3DLayer
from src.model.deeplearn.layer.downsampling_spconv3d_layer import \
    DownsamplingSpConv3DLayer
from src.model.deeplearn.layer.upsampling_spconv3d_layer import \
    UpsamplingSpConv3DLayer
from src.model.deeplearn.layer.shadow_conv1d_layer import \
    ShadowConv1DLayer
from src.model.deeplearn.layer.shadow_batch_normalization_layer import \
    ShadowBatchNormalizationLayer
from src.model.deeplearn.layer.shadow_activation_layer import \
    ShadowActivationLayer
from src.model.deeplearn.layer.sparse_indexing_map_layer import \
    SparseIndexingMapLayer
from src.utils.dl_utils import DLUtils
from src.utils.dict_utils import DictUtils
from src.main.main_config import VL3DCFG
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class SpConv3DPwiseClassif(Architecture):
    """
    :author: Alberto M. Esmoris Pena

    The sparse convolutional 3D point-wise classification model is based on the
    submanifold sparse 3D convolutions introduced by Graham et al. in
    https://arxiv.org/abs/1706.01307 and https://arxiv.org/abs/1711.10275 .
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        See :meth:`architecture.Architecture.__init__`.
        """
        # Call parent's init
        if kwargs.get('arch_name', None) is None:
            kwargs['arch_name'] = 'SpConv3D_PointWise_Classification'
        super().__init__(**kwargs)
        # Set defaults from VL3DCFG
        kwargs = DictUtils.add_defaults(
            kwargs,
            VL3DCFG['MODEL']['SpConv3DPwiseClassif']
        )
        # Assign attributes
        self.fnames = kwargs.get('fnames', None)
        if self.fnames is None:
            self.fnames = ['ones']  # If no features are given, use ones
        self.nf = len(self.fnames)  # Number of input cell-wise features
        self.num_classes = kwargs.get('num_classes', None)
        pre_kwargs = kwargs['pre_processing']
        pre_kwargs['num_classes'] = self.num_classes
        self.pre_runnable = HierarchicalSGPreProcessorPP(**pre_kwargs)
        self.post_runnable = HierarchicalSGPostProcessorPP(self.pre_runnable)
        self.submanifold_window = self.pre_runnable.submanifold_window
        self.submanifold_filters = kwargs['submanifold_filters']
        self.submanifold_features = kwargs['submanifold_features']
        self.submanifold_initializer = kwargs['submanifold_initializer']
        self.submanifold_regularizer = kwargs['submanifold_regularizer']
        self.submanifold_constraint = kwargs['submanifold_constraint']
        self.submanifold_bn_momentum = kwargs['submanifold_bn_momentum']
        self.downsampling_window = self.pre_runnable.downsampling_window
        self.downsampling_stride = self.pre_runnable.downsampling_stride
        self.downsampling_initializer = kwargs['downsampling_initializer']
        self.downsampling_regularizer = kwargs['downsampling_regularizer']
        self.downsampling_constraint = kwargs['downsampling_constraint']
        self.downsampling_bn_momentum = kwargs['downsampling_bn_momentum']
        self.upsampling_window = self.pre_runnable.upsampling_window
        self.upsampling_stride = self.pre_runnable.upsampling_stride
        self.upsampling_initializer = kwargs['upsampling_initializer']
        self.upsampling_regularizer = kwargs['upsampling_regularizer']
        self.upsampling_constraint = kwargs['upsampling_constraint']
        self.upsampling_bn_momentum = kwargs['upsampling_bn_momentum']
        self.upsampling_shared_mlp_initializer = \
            kwargs['upsampling_shared_mlp_initializer']
        self.upsampling_shared_mlp_regularizer = \
            kwargs['upsampling_shared_mlp_regularizer']
        self.upsampling_shared_mlp_constraint = \
            kwargs['upsampling_shared_mlp_constraint']
        self.upsampling_shared_mlp_activation = \
            kwargs['upsampling_shared_mlp_activation']
        self.upsampling_shared_mlp_bn_momentum = \
            kwargs['upsampling_shared_mlp_bn_momentum']
        self.feature_dim_divisor = kwargs['feature_dim_divisor']
        self.dim_transform_kernel_initializer = \
            kwargs['dim_transform_kernel_initializer']
        self.dim_transform_kernel_regularizer = \
            kwargs['dim_transform_kernel_regularizer']
        self.dim_transform_kernel_constraint = \
            kwargs['dim_transform_kernel_constraint']
        self.dim_transform_activation = kwargs['dim_transform_activation']
        self.dim_transform_bn_momentum = kwargs['dim_transform_bn_momentum']
        self.residual_strategy = kwargs['residual_strategy']
        self.residual_shared_mlp_after_ssc3d = \
            kwargs['residual_shared_mlp_after_ssc3d']
        self.residual_shared_mlp_kernel_initializer = \
            kwargs['residual_shared_mlp_kernel_initializer']
        self.residual_shared_mlp_kernel_regularizer = \
            kwargs['residual_shared_mlp_kernel_regularizer']
        self.residual_shared_mlp_kernel_constraint = \
            kwargs['residual_shared_mlp_kernel_constraint']
        self.residual_shared_mlp_activation = \
            kwargs['residual_shared_mlp_activation']
        self.initial_shared_mlp = kwargs['initial_shared_mlp']
        self.initial_shared_mlp_initializer = \
            kwargs['initial_shared_mlp_initializer']
        self.initial_shared_mlp_regularizer = \
            kwargs['initial_shared_mlp_regularizer']
        self.initial_shared_mlp_constraint = \
            kwargs['initial_shared_mlp_constraint']
        self.initial_shared_mlp_activation = \
            kwargs['initial_shared_mlp_activation']
        self.initial_unactivated_spconv = kwargs['initial_unactivated_spconv']
        self.output_kernel_initializer = kwargs['output_kernel_initializer']
        self.output_kernel_regularizer = kwargs['output_kernel_regularizer']
        self.output_kernel_constraint = kwargs['output_kernel_constraint']
        self.max_depth = len(self.submanifold_window)
        self.binary_crossentropy = False
        comp_args = kwargs.get('compilation_args', None)
        self.binary_crossentropy = DLUtils.is_using_binary_crossentropy(
            comp_args, default=False
        )
        # Cache-like attributes
        self.F = None
        self.hk, self.hv = None, None
        self.hD, self.hU = None, None
        self.n = None
        self.start = None
        self.siml = None
        self.skip_links = None

    # ---   ARCHITECTURE METHODS   --- #
    # -------------------------------- #
    def build_input(self):
        r"""
        Build the input layer of the neural network. A submanifold sparse
        3D convolutional point-wise classifier must receive many input tensors
        representing the hierarchical nature of the architecture. More
        concretely, for each element in the batch there must be:

        1) The input matrix of :math:`n_f \in \mathbb{Z}_{>0}` features that
            represents the feature space encoded for the first sparse grid in
            the hierarchy (where :math:`R_1` is the number of active cells
            at the first sparse grid in the hierarchy):

            .. math::
                \pmb{F} \in \mathbb{R_1 \times n_f}

        2) The keys of the submanifold maps :math:`h_1, \ldots, h_{t^*}`
            as ragged tensors (up to the max depth
            :math:`t^* \in \mathbb{Z}_{>0}`).

        3) The values of the submanifold maps :math:`h_1, \ldots, h_{t^*}`
            as ragged tensors (up to the max depth
            :math:`t^* \in \mathbb{Z}_{>0}`).

        4) The downsampling vectors representing the indices of the active cells
            in the corresponding non-downsampled sparse grids
            :math:`h^D_1, \ldots, h^D_{t^*}` as ragged tensors. These active
            cells are understood as the min vertex for the downsampling
            convolutional windows.

        5) The upsampling vectors representing the indices of the active cells
            in the corresponding non-upsampled sparse grids
            :math:`h^U_1, \ldots, h^U_{t^*}` as ragged tensors. These active
            cells are understood as the max vertex for the upsampling
            convolutional windows.

        6) The number of axis-wise partitions for each sparse grid
            :math:`\pmb{N} \in \mathbb{Z}^{3 \times t^*}` such that
            :math:`\pmb{n_{*t}} \in \mathbb{Z}^{3}` gives the number of
            partitions along each axis of the sparse grid at depth :math:`t`.
            For the sake of convenience, the axis-wise partitions are given
            as :math:`t^*` vectors instead of as a whole matrix.

        :return: Built layers.
        :rtype: list of :class:`tf.RaggedTensor` and :class:`tf.Tensor`
        """
        # Handle input feature spaces
        self.F = tf.keras.layers.Input(
            shape=(None, self.nf),
            ragged=False,
            dtype='float32',
            name='Fin'
        )
        # Handle input submanifold map keys
        self.hk = [
            tf.keras.layers.Input(
                shape=(None,),
                ragged=False,
                dtype='int32',
                name=f'hk_{t+1}'
            )
                for t in range(self.max_depth)
        ]
        # Handle input submanifold map values
        self.hv = [
            tf.keras.layers.Input(
                shape=(None,),
                ragged=False,
                dtype='int32',
                name=f'hv_{t+1}'
            )
            for t in range(self.max_depth)
        ]
        # Handle input downsampling vectors
        self.hD = [
            tf.keras.layers.Input(
                shape=(None,),
                ragged=False,
                dtype='int32',
                name=f'hD_{t}'
            )
            for t in range(1, self.max_depth)
        ]
        # Handle input upsampling vectors
        self.hU = [
            tf.keras.layers.Input(
                shape=(None,),
                ragged=False,
                dtype='int32',
                name=f'hU_{t}'
            )
            for t in range(1, self.max_depth)
        ]
        # Handle input number of axis-wise partitions
        self.n = [
            tf.keras.layers.Input(
                shape=(3, ),
                ragged=False,
                dtype='int32',
                name=f'n_{t+1}'
            )
            for t in range(self.max_depth)
        ]
        # Handle start indices for each element at each depth
        self.start = [
            tf.keras.layers.Input(
                shape=(1, ),
                ragged=False,
                dtype='int32',
                name=f'start_{t+1}'
            )
            for t in range(self.max_depth)
        ]
        # Return list of inputs
        return [
            self.F, *self.hk, *self.hv, *self.hD, *self.hU, *self.n, *self.start
        ]

    def build_hidden(self, x, **kwargs):
        """
        Build the hidden layers of the submanifold sparse 3D convolutional
        point-wise classifier.

        :return: The last hidden layer.
        :rtype: :class:`tf.RaggedTensor`
        """
        # Build sparse indexing map layers
        F = self.F
        self.siml = []
        for t in range(self.max_depth):
            siml = SparseIndexingMapLayer(name=f"SIML_t{t+1}")
            F = siml([self.hk[t], self.hv[t], F])
            self.siml.append(siml)
        # Build initial shared MLP (if requested)
        if self.initial_shared_mlp:
            F = ShadowConv1DLayer(
                self.nf,
                1,
                kernel_initializer=self.initial_shared_mlp_initializer,
                kernel_regularizer=self.initial_shared_mlp_regularizer,
                kernel_constraint=self.initial_shared_mlp_constraint,
                activation=self.initial_shared_mlp_activation,
                offset=1,
                name='INIT_SharedMLP'
            )([F, self.start[0]])
        # Build unactivated submanifold sparse convolution (if requested)
        if self.initial_unactivated_spconv:
            F = SubmanifoldSpConv3DLayer(
                self.submanifold_window[0],
                self.submanifold_filters[0],
                self.nf,
                self.submanifold_features[0],
                W_initializer=self.submanifold_initializer[0],
                W_regularizer=self.submanifold_regularizer[0],
                W_constraint=self.submanifold_constraint[0],
                siml=self.siml[0],
                name='UNACT_SSC3D'
            )([F, self.hk[0], self.n[0], self.start[0]])
        # Build hierarchy
        return self.build_spconv_hierarchy(F)

    def build_output(self, x, **kwargs):
        """
        Build the output layer of the submanifold sparse 3D convolutional
        point-wise classifier.

        See :meth:`architecture.Architecture.build_output`.
        """
        # Handle output normalization
        x = ShadowBatchNormalizationLayer(
            momentum=self.submanifold_bn_momentum[0],
            offset=1,
            name=f'out_BN'
        )([x, self.start[0]])
        x = ShadowActivationLayer(
            tf.keras.activations.relu,
            offset=1,
            name=f'out_ReLU'
        )([x, self.start[0]])
        # Handle output layer for binary cross-entropy loss
        if self.binary_crossentropy:
            return ShadowConv1DLayer(
                1,
                kernel_size=1,
                activation='sigmoid',
                kernel_initializer=self.output_kernel_initializer,
                kernel_regularizer=self.output_kernel_regularizer,
                kernel_constraint=self.output_kernel_constraint,
                offset=1,
                name='cwise_out'
            )([x, self.start[0]])
        # Handle output layer for the general case
        return ShadowConv1DLayer(
            self.num_classes,
            kernel_size=1,
            activation='softmax',
            kernel_initializer=self.output_kernel_initializer,
            kernel_regularizer=self.output_kernel_regularizer,
            kernel_constraint=self.output_kernel_constraint,
            offset=1,
            name='cwise_out'
        )([x, self.start[0]])

    # ---  SPCONV HIERARCHY METHODS  --- #
    # ---------------------------------- #
    def build_spconv_hierarchy(self, F):
        """
        Build the sparse convolutional hierarchy on the given input features.
        :param F: The input features
        :return: The last hidden layer of the hierarchy.
        :rtype: :class:`tf.RaggedTensor`
        """
        # Build downsampling hierarchy
        x = F
        nf = [self.nf] + self.submanifold_features
        max_depth_minus_one = self.max_depth - 1
        self.skip_links = []
        residual_strategy = self.residual_strategy
        if isinstance(residual_strategy, str):
            residual_strategy = residual_strategy.lower()
        for t in range(self.max_depth):
            # Build SpConv block
            if residual_strategy is None or residual_strategy == "null":
                x = self.build_nonresidual_spconv_block(t, nf, x, 'SSC3DD')
            elif residual_strategy == 'ssc3d':
                x = self.build_residual_spconv_block(t, nf, x, 'SSC3DD')
            elif(
                    residual_strategy == 'conv1d' or
                    residual_strategy == 'sharedmlp'
            ):
                x = self.build_residual_conv1d_block(t, nf, x, 'SSC3DD')
            else:
                raise DeepLearningException(
                    'SpConv3DPwiseClassif cannot build SpConv hierarchy with '
                    f'"{self.residual_strategy}" residual strategy.'
                )
            # Handle skip link and downsampling
            if t < max_depth_minus_one:
                self.skip_links.append(x)
                x = ShadowBatchNormalizationLayer(
                    momentum=self.downsampling_bn_momentum[t],
                    offset=1,
                    name=f'DOWN_SSC3D_BN_t{t + 1}'
                )([x, self.start[t]])
                x = ShadowActivationLayer(
                    tf.keras.activations.relu,
                    offset=1,
                    name=f'DOWN_SSC3D_ReLU_t{t + 1}'
                )([x, self.start[t]])
                x = DownsamplingSpConv3DLayer(
                    self.downsampling_window[t],
                    self.submanifold_filters[t+1],
                    nf[t+1],
                    self.submanifold_features[t+1],
                    W_initializer=self.downsampling_initializer[t],
                    W_regularizer=self.downsampling_regularizer[t],
                    W_constraint=self.downsampling_constraint[t],
                    siml=self.siml[t],
                    name=f'DOWN_SSC3D_t{t+1}'
                )([
                    x, self.hD[t], self.n[t], self.start[t+1], self.start[t]
                ])
        # Build upsampling hierarchy
        for t in range(self.max_depth-1, 0, -1):
            # Handle upsampling
            x = ShadowBatchNormalizationLayer(
                momentum=self.upsampling_bn_momentum[t-1],
                offset=1,
                name=f'UP_SSC3D_BN_t{t}'
            )([x, self.start[t]])
            x = ShadowActivationLayer(
                tf.keras.activations.relu,
                offset=1,
                name=f'UP_SSC3D_ReLU_t{t}'
            )([x, self.start[t]])
            x = UpsamplingSpConv3DLayer(
                self.upsampling_window[t-1],
                self.submanifold_filters[t-1],
                nf[t+1],
                self.submanifold_features[t-1],
                W_initializer=self.upsampling_initializer[t-1],
                W_regularizer=self.upsampling_regularizer[t-1],
                W_constraint=self.upsampling_constraint[t-1],
                siml=self.siml[t],
                name=f'UP_SCC3D_t{t}'
            )([
                x, self.hU[t-1], self.n[t], self.start[t-1], self.start[t]
            ])
            # Handle skiplink
            skip_link = self.skip_links[t-1]
            x = tf.keras.layers.Concatenate(
                name=f'UP_CONCAT_t{t}'
            )([x, skip_link])
            x = ShadowBatchNormalizationLayer(
                momentum=self.upsampling_shared_mlp_bn_momentum[t-1],
                offset=1,
                name=f'UP_SharedMLP_BN_t{t}'
            )([x, self.start[t-1]])
            x = ShadowActivationLayer(
                tf.keras.activations.relu,
                offset=1,
                name=f'UP_SharedMLP_ReLU_t{t}'
            )([x, self.start[t-1]])
            x = ShadowConv1DLayer(
                self.submanifold_features[t-1],
                1,
                kernel_initializer=self.upsampling_shared_mlp_initializer[t-1],
                kernel_regularizer=self.upsampling_shared_mlp_regularizer[t-1],
                kernel_constraint=self.upsampling_shared_mlp_constraint[t-1],
                activation=self.upsampling_shared_mlp_activation[t-1],
                offset=1,
                name=f'UP_SharedMLP_t{t}'
            )([x, self.start[t-1]])
            # Further sparse convolutions
            if t > 1:
                # Build SpConv block
                if residual_strategy is None or residual_strategy == "null":
                    x = self.build_nonresidual_spconv_block(
                        t-1, nf, x, 'SSC3DU'
                    )
                elif residual_strategy == 'ssc3d':
                    x = self.build_residual_spconv_block(t-1, nf, x, 'SSC3DU')
                elif(
                        residual_strategy == 'conv1d' or
                        residual_strategy == 'sharedmlp'
                ):
                    x = self.build_residual_conv1d_block(t-1, nf, x, 'SSC3DU')
                else:
                    raise DeepLearningException(
                        'SpConv3DPwiseClassif cannot build SpConv hierarchy '
                        f'with "{self.residual_strategy}" residual strategy.'
                    )
        # Return last hidden layer
        return x

    def build_nonresidual_spconv_block(self, t, nf, x, infix):
        """
        Build a fully sequential sparse convolutional block, i.e., without
        parallel downstream.

        :return: The final tensor at the end of the non-residual SpConv block.
        :rtype: :class:`tf.Tensor`
        """
        # Build first SSC3D block
        x = ShadowBatchNormalizationLayer(
            momentum=self.submanifold_bn_momentum[t],
            offset=1,
            name=f'PREACT_{infix}1_BN_t{t+1}'
        )([x, self.start[t]])
        x = ShadowActivationLayer(
            tf.keras.activations.relu,
            offset=1,
            name=f'PREACT_{infix}1_ReLU_t{t+1}'
        )([x, self.start[t]])
        Din, DinTrans, Dout, DoutTrans, x = self.spconv_prewrap(
            nf[t+1],  # Din
            self.submanifold_features[t],  # Dout
            x,  # Current input
            t,  # Current depth
            self.start[t],  # Current start row-index
            name_prefix=f'{infix}1_PreSharedMLP_'  # Prefix for layer names
        )
        x = SubmanifoldSpConv3DLayer(
            self.submanifold_window[t],
            self.submanifold_filters[t],
            DinTrans,
            DoutTrans,
            W_initializer=self.submanifold_initializer[t],
            W_regularizer=self.submanifold_regularizer[t],
            W_constraint=self.submanifold_constraint[t],
            siml=self.siml[t],
            name=f'PREACT_{infix}1_t{t+1}'
        )([x, self.hk[t], self.n[t], self.start[t]])
        # Build second SSC3D block
        x = ShadowBatchNormalizationLayer(
            momentum=self.submanifold_bn_momentum[t],
            offset=1,
            name=f'PREACT_{infix}2_BN_t{t+1}'
        )([x, self.start[t]])
        x = ShadowActivationLayer(
            tf.keras.activations.relu,
            offset=1,
            name=f'PREACT_{infix}2_ReLU_t{t+1}'
        )([x, self.start[t]])
        x = SubmanifoldSpConv3DLayer(
            self.submanifold_window[t],
            self.submanifold_filters[t],
            DoutTrans,
            DoutTrans,
            W_initializer=self.submanifold_initializer[t],
            W_regularizer=self.submanifold_regularizer[t],
            W_constraint=self.submanifold_constraint[t],
            siml=self.siml[t],
            name=f'PREACT_{infix}2_t{t+1}'
        )([x, self.hk[t], self.n[t], self.start[t]])
        x = self.spconv_postwrap(
            Dout,  # Dout
            x,  # Current input
            t,  # Current depth
            self.start[t],  # Current start row-index
            f'{infix}2_PostSharedMLP_'  # Prefix for layer names
        )
        # Return non-residual SpConv block
        return x

    def build_residual_spconv_block(self, t, nf, x, infix):
        """
        Build a fully sequential sparse convolutional block with residual, i.e.,
        including a parallel downstream with an alternative sparse convolution.

        :return: The final tensor at the end of the residual SpConv block.
        :rtype: :class:`tf.Tensor`
        """
        # Common input processing before residual branching
        x = ShadowBatchNormalizationLayer(
            momentum=self.submanifold_bn_momentum[t],
            offset=1,
            name=f'PREACT_{infix}1_BN_t{t+1}'
        )([x, self.start[t]])
        x = ShadowActivationLayer(
            tf.keras.activations.relu,
            offset=1,
            name=f'PREACT_{infix}1_ReLU_t{t+1}'
        )([x, self.start[t]])
        _feature_dim_divisor = self.feature_dim_divisor
        self.feature_dim_divisor = self.feature_dim_divisor * 2
        Din, DinTrans, Dout, DoutTrans, x = self.spconv_prewrap(
            nf[t+1],  # Din
            self.submanifold_features[t],  # Dout
            x,  # Current input
            t,  # Current depth
            self.start[t],  # Current start row-index
            name_prefix=f'{infix}1_PreSharedMLP_'  # Prefix for layer names
        )
        # Build residual branch
        xres = SubmanifoldSpConv3DLayer(
            self.submanifold_window[t],
            self.submanifold_filters[t],
            DinTrans,
            DoutTrans,
            W_initializer=self.submanifold_initializer[t],
            W_regularizer=self.submanifold_regularizer[t],
            W_constraint=self.submanifold_constraint[t],
            siml=self.siml[t],
            name=f'RES_{infix}1_t{t+1}'
        )([x, self.hk[t], self.n[t], self.start[t]])
        if self.residual_shared_mlp_after_ssc3d:
            xres = ShadowBatchNormalizationLayer(
                momentum=self.submanifold_bn_momentum[t],
                offset=1,
                name=f'RES_{infix}_SharedMLP_BN_t{t+1}'
            )([xres, self.start[t]])
            xres = ShadowActivationLayer(
                tf.keras.activations.relu,
                offset=1,
                name=f'RES_{infix}_SharedMLP_ReLU_t{t+1}'
            )([xres, self.start[t]])
            xres = ShadowConv1DLayer(
                DoutTrans,
                1,
                kernel_initializer=self.residual_shared_mlp_kernel_initializer,
                kernel_regularizer=self.residual_shared_mlp_kernel_regularizer,
                kernel_constraint=self.residual_shared_mlp_kernel_constraint,
                activation=self.residual_shared_mlp_activation,
                offset=1,
                name=f'RES_{infix}_SharedMLP_t{t+1}'
            )([xres, self.start[t]])
        # Build non-residual branch : first SSC3D block
        x = SubmanifoldSpConv3DLayer(
            self.submanifold_window[t],
            self.submanifold_filters[t],
            DinTrans,
            DoutTrans,
            W_initializer=self.submanifold_initializer[t],
            W_regularizer=self.submanifold_regularizer[t],
            W_constraint=self.submanifold_constraint[t],
            siml=self.siml[t],
            name=f'PREACT_{infix}1_t{t+1}'
        )([x, self.hk[t], self.n[t], self.start[t]])
        # Build non-residual branch: second SSC3D block
        x = ShadowBatchNormalizationLayer(
            momentum=self.submanifold_bn_momentum[t],
            offset=1,
            name=f'PREACT_{infix}2_BN_t{t+1}'
        )([x, self.start[t]])
        x = ShadowActivationLayer(  # TOOD Restore : Debug
            tf.keras.activations.relu,
            offset=1,
            name=f'PREACT_{infix}2_ReLU_t{t+1}'
        )([x, self.start[t]])
        x = SubmanifoldSpConv3DLayer(
            self.submanifold_window[t],
            self.submanifold_filters[t],
            DoutTrans,
            DoutTrans,
            W_initializer=self.submanifold_initializer[t],
            W_regularizer=self.submanifold_regularizer[t],
            W_constraint=self.submanifold_constraint[t],
            siml=self.siml[t],
            name=f'PREACT_{infix}2_t{t+1}'
        )([x, self.hk[t], self.n[t], self.start[t]])
        # Concatenate residual and non-residual branches
        x = tf.keras.layers.Concatenate(
            name=f'RES_{infix}_CONCAT_t{t+1}'
        )([x, xres])
        # Do post-wrap
        x = self.spconv_postwrap(
            Dout,  # Dout
            x,  # Current input
            t,  # Current depth
            self.start[t],  # Current start row-index
            f'{infix}2_PostSharedMLP_'  # Prefix for layer names
        )
        self.feature_dim_divisor= _feature_dim_divisor
        # Return residual SpConv block
        return x

    def build_residual_conv1d_block(self, t, nf, x, infix):
        """
        Build a fully sequential sparse convolutional block with residual, i.e.,
        including a parallel downstream with an alternative Shared MLP.

        :return: The final tensor at the end of the residual SpConv block.
        :rtype: :class:`tf.Tensor`
        """
        # Common input processing before residual branching
        x = ShadowBatchNormalizationLayer(
            momentum=self.submanifold_bn_momentum[t],
            offset=1,
            name=f'PREACT_{infix}1_BN_t{t+1}'
        )([x, self.start[t]])
        x = ShadowActivationLayer(
            tf.keras.activations.relu,
            offset=1,
            name=f'PREACT_{infix}1_ReLU_t{t+1}'
        )([x, self.start[t]])
        _feature_dim_divisor = self.feature_dim_divisor
        self.feature_dim_divisor = self.feature_dim_divisor * 2
        Din, DinTrans, Dout, DoutTrans, x = self.spconv_prewrap(
            nf[t+1],  # Din
            self.submanifold_features[t],  # Dout
            x,  # Current input
            t,  # Current depth
            self.start[t],  # Current start row-index
            name_prefix=f'{infix}1_PreSharedMLP_'  # Prefix for layer names
        )
        # Build residual branch
        xres = ShadowConv1DLayer(
            DoutTrans,
            1,
            kernel_initializer=self.residual_shared_mlp_kernel_initializer,
            kernel_regularizer=self.residual_shared_mlp_kernel_regularizer,
            kernel_constraint=self.residual_shared_mlp_kernel_constraint,
            activation=self.residual_shared_mlp_activation,
            offset=1,
            name=f'RES_{infix}_SharedMLP_t{t+1}'
        )([x, self.start[t]])
        # Build non-residual branch : first SSC3D block
        x = SubmanifoldSpConv3DLayer(
            self.submanifold_window[t],
            self.submanifold_filters[t],
            DinTrans,
            DoutTrans,
            W_initializer=self.submanifold_initializer[t],
            W_regularizer=self.submanifold_regularizer[t],
            W_constraint=self.submanifold_constraint[t],
            siml=self.siml[t],
            name=f'PREACT_{infix}1_t{t+1}'
        )([x, self.hk[t], self.n[t], self.start[t]])
        # Build non-residual branch: second SSC3D block
        x = ShadowBatchNormalizationLayer(
            momentum=self.submanifold_bn_momentum[t],
            offset=1,
            name=f'PREACT_{infix}2_BN_t{t+1}'
        )([x, self.start[t]])
        x = ShadowActivationLayer(
            tf.keras.activations.relu,
            offset=1,
            name=f'PREACT_{infix}2_ReLU_t{t+1}'
        )([x, self.start[t]])
        x = SubmanifoldSpConv3DLayer(
            self.submanifold_window[t],
            self.submanifold_filters[t],
            DoutTrans,
            DoutTrans,
            W_initializer=self.submanifold_initializer[t],
            W_regularizer=self.submanifold_regularizer[t],
            W_constraint=self.submanifold_constraint[t],
            siml=self.siml[t],
            name=f'PREACT_{infix}2_t{t+1}'
        )([x, self.hk[t], self.n[t], self.start[t]])
        # Concatenate residual and non-residual branches
        x = tf.keras.layers.Concatenate(
            name=f'RES_{infix}_CONCAT_t{t+1}'
        )([x, xres])
        # Do post-wrap
        x = self.spconv_postwrap(
            Dout,  # Dout
            x,  # Current input
            t,  # Current depth
            self.start[t],  # Current start row-index
            f'{infix}2_PostSharedMLP_'  # Prefix for layer names
        )
        self.feature_dim_divisor= _feature_dim_divisor
        # Return residual SpConv block
        return x

    # ---  WRAPPER BLOCKS  --- #
    # ------------------------ #
    def spconv_prewrap(self, Din, Dout, x, t, start, name_prefix):
        """
        Transform the input feature space and determine the corresponding
        input and output transformed dimensionalities. Note that if feature
        dimensionality divisor (self.feature_dim_divisor) is zero or one the
        prewrap block will not change anything.

        :param Din: Original input dimensionality for the feature space.
        :param Dout: Original output dimensionality for the feature space.
        :param x: The input feature space.
        :param t: The current depth.
        :param start: The start row-index to handle the padding in the feature
            space.
        :param name_prefix: The prefix for the names of the layers.
        :return: The input dimensionality, the transformed input dimensionality,
            the output dimensionality, the transformed output dimensionality,
            and the transformed input feature space.
        :rtype: tuple
        """
        # Check a prewrap transform is needed
        if(
            self.feature_dim_divisor is None or
            self.feature_dim_divisor == 0 or
            self.feature_dim_divisor == 1
        ):
            return Din, Din, Dout, Dout, x
        # If so, compute the transformed input and output dimensionalities
        DinTrans = Din // self.feature_dim_divisor
        DoutTrans = Dout // self.feature_dim_divisor
        # Also, transform the input feature space
        x = ShadowConv1DLayer(
            DinTrans,
            1,
            kernel_initializer=self.dim_transform_kernel_initializer,
            kernel_regularizer=self.dim_transform_kernel_regularizer,
            kernel_constraint=self.dim_transform_kernel_constraint,
            activation=self.dim_transform_activation,
            offset=1,
            name=f'{name_prefix}t{t+1}'
        )([x, start])
        x = ShadowBatchNormalizationLayer(
            momentum=self.dim_transform_bn_momentum,
            offset=1,
            name=f'{name_prefix}BN_t{t+1}'
        )([x, start])
        x = ShadowActivationLayer(
            tf.keras.activations.relu,
            offset=1,
            name=f'{name_prefix}ReLU_t{t+1}'
        )([x, start])
        # Return transformed dimensionalities and features
        return Din, DinTrans, Dout, DoutTrans, x

    def spconv_postwrap(self, Dout, x, t, start, name_prefix):
        """
        Transform the output feature space to its final dimensionality. Note
        that if feature dimensionality divisor (self.feature_dim_divisor) is
        zero or one the postwrap block will not change anything.

        :param Dout: Original output dimensionality for the feature space (not
            necessarily the current one, but the final one).
        :param x: The output feature space.
        :param t: The current depth.
        :param start: The start row-index to handle the padding in the feature
            space.
        :param name_prefix: The prefix for the names of the layers.
        :return: The transformed output feature space.
        :rtype: :class:`tf.Tensor`
        """
        # Check a postwrap transform is needed
        if(
                self.feature_dim_divisor is None or
                self.feature_dim_divisor == 0 or
                self.feature_dim_divisor == 1
        ):
            return x
        # Transform output feature space
        x = ShadowBatchNormalizationLayer(
            momentum=self.dim_transform_bn_momentum,
            offset=1,
            name=f'{name_prefix}BN_t{t+1}'
        )([x, start])
        x = ShadowActivationLayer(  # TOOD Restore : Debug
            tf.keras.activations.relu,
            offset=1,
            name=f'{name_prefix}ReLU_t{t+1}'
        )([x, start])
        x = ShadowConv1DLayer(
            Dout,
            1,
            kernel_initializer=self.dim_transform_kernel_initializer,
            kernel_regularizer=self.dim_transform_kernel_regularizer,
            kernel_constraint=self.dim_transform_kernel_constraint,
            activation=self.dim_transform_activation,
            offset=1,
            name=f'{name_prefix}t{t+1}'
        )([x, start])
        # Return transformed output feature space
        return x

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def __getstate__(self):
        """
        Method to be called when saving the serialized SpConv3DPwiseClassif
        architecture.

        :return: The state's dictionary of the object.
        :rtype: dict
        """
        # Call parent's method
        state = super().__getstate__()
        # Add SpConv3DPwiseClassif's attributes to state dictionary
        state['fnames'] = self.fnames
        state['nf'] = self.nf
        state['num_classes'] = self.num_classes
        state['submanifold_window'] = self.submanifold_window
        state['submanifold_filters'] = self.submanifold_filters
        state['submanifold_features'] = self.submanifold_features
        state['submanifold_initializer'] = self.submanifold_initializer
        state['submanifold_regularizer'] = self.submanifold_regularizer
        state['submanifold_constraint'] = self.submanifold_constraint
        state['submanifold_bn_momentum'] = self.submanifold_bn_momentum
        state['downsampling_window'] = self.downsampling_window
        state['downsampling_stride'] = self.downsampling_stride
        state['downsampling_initializer'] = self.downsampling_initializer
        state['downsampling_regularizer'] = self.downsampling_regularizer
        state['downsampling_constraint'] = self.downsampling_constraint
        state['downsampling_bn_momentum'] = self.downsampling_bn_momentum
        state['upsampling_window'] = self.upsampling_window
        state['upsampling_stride'] = self.upsampling_stride
        state['upsampling_initializer'] = self.upsampling_initializer
        state['upsampling_regularizer'] = self.upsampling_regularizer
        state['upsampling_constraint'] = self.upsampling_constraint
        state['upsampling_bn_momentum'] = self.upsampling_bn_momentum
        state['upsampling_shared_mlp_initializer'] = \
            self.upsampling_shared_mlp_initializer
        state['upsampling_shared_mlp_regularizer'] = \
            self.upsampling_shared_mlp_regularizer
        state['upsampling_shared_mlp_constraint'] = \
            self.upsampling_shared_mlp_constraint
        state['upsampling_shared_mlp_activation'] = \
            self.upsampling_shared_mlp_activation
        state['upsampling_shared_mlp_bn_momentum'] = \
            self.upsampling_shared_mlp_bn_momentum
        state['feature_dim_divisor'] = self.feature_dim_divisor
        state['dim_transform_kernel_initializer'] = \
            self.dim_transform_kernel_initializer
        state['dim_transform_kernel_regularizer'] = \
            self.dim_transform_kernel_regularizer
        state['dim_transform_kernel_constraint'] = \
            self.dim_transform_kernel_constraint
        state['dim_transform_activation'] = self.dim_transform_activation
        state['dim_transform_bn_momentum'] = self.dim_transform_bn_momentum
        state['residual_strategy'] = self.residual_strategy
        state['residual_shared_mlp_after_ssc3d'] = \
            self.residual_shared_mlp_after_ssc3d
        state['residual_shared_mlp_kernel_initializer'] = \
            self.residual_shared_mlp_kernel_initializer
        state['residual_shared_mlp_kernel_regularizer'] = \
            self.residual_shared_mlp_kernel_regularizer
        state['residual_shared_mlp_kernel_constraint'] = \
            self.residual_shared_mlp_kernel_constraint
        state['residual_shared_mlp_activation'] = \
            self.residual_shared_mlp_activation
        state['initial_shared_mlp'] = self.initial_shared_mlp
        state['initial_shared_mlp_initializer'] = \
            self.initial_shared_mlp_initializer
        state['initial_shared_mlp_regularizer'] = \
            self.initial_shared_mlp_regularizer
        state['initial_shared_mlp_constraint'] = \
            self.initial_shared_mlp_constraint
        state['initial_shared_mlp_activation'] = \
            self.initial_shared_mlp_activation
        state['initial_unactivated_spconv'] = self.initial_unactivated_spconv
        state['output_kernel_initializer'] = self.output_kernel_initializer
        state['output_kernel_regularizer'] = self.output_kernel_regularizer
        state['output_kernel_constraint'] = self.output_kernel_constraint
        state['max_depth'] = self.max_depth
        # Return
        return state

    def __setstate__(self, state):
        """
        Method to be called when loading and deserializing a previously
        serialized SpConv3DPwiseClassif architecture.

        :param state: The state's dictionary of the saved
            SpConv3DPwiseClassif architecture.
        :type state: dict
        :return: Nothing, but modifies the internal state of the object.
        """
        # Assign SpConv3DPwiseClassif's attributes from state dictionary
        self.fnames = state['fnames']
        self.nf = state['nf']
        self.num_classes = state['num_classes']
        self.submanifold_window = state['submanifold_window']
        self.submanifold_filters = state['submanifold_filters']
        self.submanifold_features = state['submanifold_features']
        self.submanifold_initializer = state['submanifold_initializer']
        self.submanifold_regularizer = state['submanifold_regularizer']
        self.submanifold_constraint = state['submanifold_constraint']
        self.submanifold_bn_momentum = state['submanifold_bn_momentum']
        self.downsampling_window = state['downsampling_window']
        self.downsampling_stride = state['downsampling_stride']
        self.downsampling_initializer = state['downsampling_initializer']
        self.downsampling_regularizer = state['downsampling_regularizer']
        self.downsampling_constraint = state['downsampling_constraint']
        self.downsampling_bn_momentum = state['downsampling_bn_momentum']
        self.upsampling_window = state['upsampling_window']
        self.upsampling_stride = state['upsampling_stride']
        self.upsampling_initializer = state['upsampling_initializer']
        self.upsampling_regularizer = state['upsampling_regularizer']
        self.upsampling_constraint = state['upsampling_constraint']
        self.upsampling_bn_momentum = state['upsampling_bn_momentum']
        self.upsampling_shared_mlp_initializer = \
            state['upsampling_shared_mlp_initializer']
        self.upsampling_shared_mlp_regularizer = \
            state['upsampling_shared_mlp_regularizer']
        self.upsampling_shared_mlp_constraint = \
            state['upsampling_shared_mlp_constraint']
        self.upsampling_shared_mlp_activation = \
            state['upsampling_shared_mlp_activation']
        self.upsampling_shared_mlp_bn_momentum = \
            state['upsampling_shared_mlp_bn_momentum']
        self.feature_dim_divisor = state['feature_dim_divisor']
        self.dim_transform_kernel_initializer = \
            state['dim_transform_kernel_initializer']
        self.dim_transform_kernel_regularizer = \
            state['dim_transform_kernel_regularizer']
        self.dim_transform_kernel_constraint = \
            state['dim_transform_kernel_constraint']
        self.dim_transform_activation = state['dim_transform_activation']
        self.dim_transform_bn_momentum = state['dim_transform_bn_momentum']
        self.residual_strategy = state['residual_strategy']
        self.residual_shared_mlp_after_ssc3d = \
            state['residual_shared_mlp_after_ssc3d']
        self.residual_shared_mlp_kernel_initializer = \
            state['residual_shared_mlp_kernel_initializer']
        self.residual_shared_mlp_kernel_regularizer = \
            state['residual_shared_mlp_kernel_regularizer']
        self.residual_shared_mlp_kernel_constraint = \
            state['residual_shared_mlp_kernel_constraint']
        self.residual_shared_mlp_activation = \
            state['residual_shared_mlp_activation']
        self.initial_shared_mlp = state['initial_shared_mlp']
        self.initial_shared_mlp_initializer = \
            state['initial_shared_mlp_initializer']
        self.initial_shared_mlp_regularizer = \
            state['initial_shared_mlp_regularizer']
        self.initial_shared_mlp_constraint = \
            state['initial_shared_mlp_constraint']
        self.initial_shared_mlp_activation = \
            state['initial_shared_mlp_activation']
        self.initial_unactivated_spconv = state['initial_unactivated_spconv']
        self.output_kernel_initializer = state['output_kernel_initializer']
        self.output_kernel_regularizer = state['output_kernel_regularizer']
        self.output_kernel_constraint = state['output_kernel_constraint']
        self.max_depth = state['max_depth']
        # Call parent's set state
        super().__setstate__(state)



