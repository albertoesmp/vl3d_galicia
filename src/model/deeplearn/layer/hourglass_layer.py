# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.layer.layer import Layer
import src.main.main_logger as LOGGING
import tensorflow as tf

# ---   CLASS   --- #
# ----------------- #
class HourglassLayer(Layer):
    r"""
    :author: Alberto M. Esmoris Pena

    An hourglass layer consists of two unbiased MLPs. The first one uses the
    weights
    :math:`\pmb{W_1} \in \mathbb{R}^{D_{\mathrm{in}} \times D_{\mathrm{h}}}`,
    to transform the features of :math:`m \in \mathbb{Z}_{>0}` points
    :math:`\pmb{F} \in \mathbb{R}^{m \times D_{\mathrm{in}}}` to a reduced
    feature space of :math:`D_{h} \in \mathbb{Z}_{>0}` features.
    Then, the second unbiased MLP transforms the features from the reduced
    space to an output feature space of
    :math:`D_{\mathrm{out}} \in \mathbb{Z}_{>0}` features using the weights
    :math:`\pmb{W_2} \in \mathbb{R}^{D_h \times D_{\mathrm{out}}}`.

    The hourglass layer is a map
    :math:`\mathcal{H} : \mathbb{R}^{m \times D_{\mathrm{in}}} \to \mathbb{R}^{m \times D_{\mathrm{out}}}`
    that can be mathematically summarized as:

    .. math::

        \pmb{Y} = \sigma_2(\sigma_1(\pmb{F} \pmb{W_1})\pmb{W_2})
            \in \mathbb{R}^{m \times D_{out}}

    Where :math:`\sigma_i` represents an activation function like a ReLU.

    A regularization strategy is necessary to circumvent any potential
    information loss when mapping a high-dimensional feature space to a
    low-dimensional feature space. This problem can be addressed adding an
    extra term to the original loss function :math:`\mathcal{L}` to obtain a
    new loss function :math:`\mathcal{L}' = \mathcal{L} + \beta \mathcal{L}_{h}`
    where the **hyperparameter** :math:`\beta \in \mathbb{R}` controls the
    contribution/impact of the hourglass regularization term to the final loss
    and:

    .. math::

        \mathcal{L}_{h} = \left\lVert{
            \dfrac{
                \pmb{W_1}^{\intercal}\pmb{W_1}
            }{
                \lVert\pmb{W_1}\rVert_2^2
            }
            - \pmb{I}
        }\right\rVert_{F}

    Where :math:`\lVert\cdot\rVert_2` is the spectral norm of a matrix and
    :math:`\lVert\cdot\rVert_F` is the Frobenius norm.

    Further information about the hourglass layer can be read in the
    SFL-NET paper (https://doi.org/10.1109/TGRS.2023.3313876).

    :ivar Dh: The requested internal dimensionality, i.e., :math:`D_{h}`.
    :vartype Dh: int
    :ivar Dout: The requested output dimensionality, i.e.,
        :math:`D_{\mathrm{out}}`.
    :vartype Dout: int
    :ivar activation: The activation function :math:`\sigma_1`.
    :vartype activation: str
    :ivar activation2: The activation function :math:`\sigma_2`.
    :vartype activation2: str
    :ivar regularize: Whether to apply :math:`+ \beta \mathcal{L}_h` to the
        loss function or not.
    :vartype regularize: bool
    :ivar beta: The loss factor :math:`\beta`.
    :ivar spectral_strategy: The type of spectral strategy to be used. It can
        be either "unsafe" (might break during training), "safe" (will not
        break during training, but it will be twice slower), or "approx"
        (as fast as unsafe but less prone to break during training).
    :vartype spectral_strategy: str
    :vartype beta: float
    :ivar built_W1: Whether the first matrix of weights
        :math:`\pmb{W_1} \in \mathbb{R}^{D_{\mathrm{in}} \times D_{h}}`
        is built or not. Initially it is false, but it will be updated once
        the layer is built.
    :vartype built_W1: bool
    :ivar built_W2: Whether the second matrix of weights
        :math:`\pmb{W_2} \in \mathbb{R}^{D_h \times D_{\mathrm{out}}}`
        is built or not. Initially it is false, but it will be updated once
        the layer is built.
    :vartype built_W2: bool
    :ivar W1_initializer: The initializer for the first matrix of weights
        :math:`\pmb{W_1} \in \mathbb{R}^{D_{\mathrm{in}} \times D_{h}}`.
    :ivar W1_regularizer: The regularizer for the first matrix of weights
        :math:`\pmb{W_1} \in \mathbb{R}^{D_{\mathrm{in}} \times D_{h}}`.
    :ivar W1_constraint: The constraint for the first matrix of weights
        :math:`\pmb{W_1} \in \mathbb{R}^{D_{\mathrm{in}} \times D_{h}}`.
    :ivar W2_initializer: The initializer for the second matrix of weights
        :math:`\pmb{W_2} \in \mathbb{R}^{D_h \times D_{\mathrm{out}}}`.
    :ivar W2_regularizer: The regularizer for the second matrix of weights
        :math:`\pmb{W_2} \in \mathbb{R}^{D_h \times D_{\mathrm{out}}}`.
    :ivar W2_constraint: The constraint for the second matrix of weights
        :math:`\pmb{W_2} \in \mathbb{R}^{D_h \times D_{\mathrm{out}}}`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(
        self,
        Dh,
        Dout,
        activation='ReLU',
        activation2='ReLU',
        regularize=True,
        spectral_strategy="approx",
        beta=0.1,
        built_W1=False,
        built_W2=False,
        W1_initializer=None,
        W1_regularizer=None,
        W1_constraint=None,
        W2_initializer=None,
        W2_regularizer=None,
        W2_constraint=None,
        sigma=None,
        sigma2=None,
        **kwargs
    ):
        """
        See :class:`.Layer` and :meth:`layer.Layer.__init__`.
        """
        # TODO Rethink : It should work without sigma (test with serialization)
        # TODO Rethink : If sigma is necessary, add to the docs.
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.Dh = Dh  # Dimensionality of internal feature space
        self.Dout = Dout  # Dimensionality of output feature space
        self.activation = activation  # Type of activation function
        self.activation2 = activation2  # Type of second activation function
        self.regularize = regularize  # Whether to apply L_h term to the loss
        self.beta = beta
        self.spectral_strategy = spectral_strategy  # Spectral norm's approach
        if self.spectral_strategy is None:
            self.spectral_strategy = "approx"
            LOGGING.LOGGER.info(
                f'HourglassLayer ({self.name}) instantiated with None spectral '
                'strategy. Note that "approx" will be used by default.'
            )
        ss_low = self.spectral_strategy.lower()
        if ss_low == 'unsafe':
            self.spectral_norm_function = self.compute_spectral_unsafe
        elif ss_low == 'safe':
            self.spectral_norm_function = self.compute_spectral_safe
        elif ss_low == 'approx':
            self.spectral_norm_function = self.compute_spectral_approx
        else:
            raise DeepLearningException(
                'Cannot instantiate HourglassLayer with the following '
                f'spectral strategy: {self.spectral_strategy}'
            )
        # Create activation function
        if sigma is None:
            alow = None if self.activation is None else self.activation.lower()
            if activation is None or alow == 'identity':
                self.sigma = lambda x: x
            elif alow == 'relu':
                self.sigma = lambda x : tf.keras.activations.relu(
                    x, alpha=0.0, max_value=None, threshold=0.0
                )
            else:
                raise DeepLearningException(
                    'HourglassLayer does not support requested activation '
                    f'function: "{self.activation}"'
                )
        else:
            self.sigma = tf.keras.activations.get(sigma)
        # Create second activation function
        if sigma2 is None:
            alow = None if self.activation2 is None else self.activation2.lower()
            if activation2 is None or alow == 'identity':
                self.sigma2 = lambda x: x
            elif alow == 'relu':
                self.sigma2 = lambda x : tf.keras.activations.relu(
                    x, alpha=0.0, max_value=None, threshold=0.0
                )
            else:
                raise DeepLearningException(
                    'HourglassLayer does not support requested second '
                    f'activation function: "{self.activation2}"'
                )
        else:
            self.sigma2 = tf.keras.activations.get(sigma2)
        # Create regularizer
        if self.regularize:
            self.regularizer = self.do_hourglass_regularization
        else:
            self.regularizer = self.do_no_regularization
        self.W1_initializer = tf.keras.initializers.get(W1_initializer)
        self.W1_regularizer = tf.keras.regularizers.get(W1_regularizer)
        self.W1_constraint = tf.keras.constraints.get(W1_constraint)
        self.W2_initializer = tf.keras.initializers.get(W2_initializer)
        self.W2_regularizer = tf.keras.regularizers.get(W2_regularizer)
        self.W2_constraint = tf.keras.constraints.get(W2_constraint)
        # Attributes initialized to None (derived when building)
        self.tikhonov_lambda = None  # Lambda for approx spectral
        self.I = None  # Identity matrix of Dh dim
        self.W1 = None  # Weights of first unbiased MLP
        self.built_W1 = built_W1  # True if built, False otherwise
        self.W2 = None  # Weights of second unbiased MLP
        self.built_W2 = built_W2  # True if built, False otherwise

    # ---   LAYER METHODS   --- #
    # ------------------------- #
    def build(self, dim_in):
        r"""
        Build the :math:`\pmb{W_1} \in \mathbb{R}^{D_{\mathrm{in}} \times D_h}`
        matrix representing the weights of the first MLP and the
        :math:`\pmb{W_2} \in \mathbb{R}^{D_h \times D_{\mathrm{out}}}` matrix
        representing the weights of the second MLP.

        See :class:`.Layer` and :meth:`layer.Layer.build`.
        """
        # Call parent's build
        super().build(dim_in)
        # Find the dimensionality of the input feature space
        Din = dim_in[-1]
        # Build the weights of the first MLP
        if not self.built_W1:
            self.W1 = self.add_weight(
                shape=(Din, self.Dh),
                initializer=self.W1_initializer,
                regularizer=self.W1_regularizer,
                constraint=self.W1_constraint,
                dtype='float32',
                trainable=True,
                name='W1'
            )
            self.built_W1 = True
        # Validate the weights of the first MLP
        if self.W1 is None:
            raise DeepLearningException(
                'HourglassLayer failed to build weights for first MLP.'
            )
        # Build the weights of the second MLP
        if not self.built_W2:
            self.W2 = self.add_weight(
                shape=(self.Dh, self.Dout),
                initializer=self.W2_initializer,
                regularizer=self.W2_regularizer,
                constraint=self.W2_constraint,
                dtype='float32',
                trainable=True,
                name='W2'
            )
            self.built_W2 = True
        # Validate the weights of the second MLP
        if self.W2 is None:
            raise DeepLearningException(
                'HourglassLayer failed to build weights for second MLP.'
            )
        # Build identity matrix of Dh dim
        self.I = tf.eye(self.Dh, dtype=self.W1.dtype)
        # Prepare Tikhonov lambda
        if self.W1.dtype == tf.float32:  # lambda for 32 bits
            self.tikhonov_lambda = 1e-5
        else:  # lambda for 64 bits
            self.tikhonov_lambda = 1e-10
        self.built = True

    def call(self, inputs, training=False, mask=False):
        r"""
        Compute:

        .. math::

            \sigma_2\left({
                \sigma_1(\pmb{F} \pmb{W_1})
                \pmb{W_2}
            }\right)

        See :class:`.HourglassLayer` for more details.

        :param inputs: The feature space tensor representing the batch of
            structure space matrices.
        :return: The output feature space tensor
            :math:`\mathcal{Y} \in \mathbb{R}^{K \times m \times D_{\mathrm{out}}}`.
            Besides, it might potentially add a term to the loss function.
        """
        # Extract input
        F = inputs
        # Compute first MLP
        Y1 = self.sigma(tf.matmul(F, self.W1))
        # Compute second MLP
        Y2 = self.sigma2(tf.matmul(Y1, self.W2))
        # Handle regularization
        self.regularizer()
        # Return final output
        return Y2


    # ---   REGULARIZATION   --- #
    # -------------------------- #
    def do_hourglass_regularization(self):
        """
        Apply the hourglass regularization described in
        :class:`.HourglassLayer`.
        """
        # Add the hourglass regularization term to the loss function
        self.add_loss(self.beta*self._do_hourglass_regularization())

    def _do_hourglass_regularization(self):
        """
        Assist the :meth:`.HourglassLayer.do_hourglass_regularization`
        in doing the computation so the former only needs to care about
        updating the loss function correctly.
        """
        # Compute covariance matrix
        cov = tf.matmul(tf.transpose(self.W1), self.W1)
        # Compute the squared spectral norm
        spectral = self.spectral_norm_function(cov)
        # Compute regularization matrix
        X = cov/spectral - self.I
        # Compute the Frobenius norm of the regularization matrix
        return tf.sqrt(tf.reduce_sum(tf.square(X)))

    def do_no_regularization(self):
        """
        Do not apply any regularization at all.
        """
        return

    def compute_spectral_unsafe(self, cov):
        """
        Because the spectral norm is computed for the W_1^T W_1 matrix, it is
        guaranteed to be hermitian and positive semidefinite. Algebraically
        speaking, its eigenvalues can be derived with the eigh routine
        that uses heevd (hermitian eigenvalue decomposition) from LAPACK.
        However, sometimes, during training, the resulting matrix might present
        numerical issues causing the heevd routine to fail.

        See :meth:`.HourglassLayer.compute_spectral_safe` and
        :meth:`.HourglassLayer.compute_spectral_approx`.
        """
        # NOTE that using tf.linalg.eigh might lead to heevd errors
        return tf.sqrt(tf.reduce_max(tf.linalg.eigh(cov)[0]))

    def compute_spectral_safe(self, cov):
        """
        A safe but two times slower alternative to the
        :meth:`.HourglassLayer.compute_spectral_unsafe` function. It uses
        the singular value decomposition approach instead of the eigh function.
        Consequently, it is robust to numerical issues and it will not fail
        during training.

        See :meth:`.HourglassLayer.compute_spectral_unsafe` and
        :meth:`.HourglassLayer.compute_spectral_approx`.
        """
        # NOTE that using tf.linalg.svd is more robust
        return tf.sqrt(tf.reduce_max(tf.linalg.svd(cov, compute_uv=False)))

    def compute_spectral_approx(self, cov):
        """
        A compromise solution that has the speed of the unsafe alternative
        but is much less likely to break the training process. It considers
        :math:`\pmb{W_1}^T \pmb{W_1} + \lambda \pmb{I}` with
        :math:`\lambda \to 0` instead of :math:`\pmb{W_1}^T \pmb{W_1}` to
        prevent numerical issues.

        See :meth:`.HourglassLayer.compute_spectral_safe` and
        :meth:`.HourglassLayer.compute_spectral_unsafe`.
        """
        # NOTE that tikhonov provides a fast yet safer approximation to eigh
        return tf.sqrt(tf.reduce_max(tf.linalg.eigh(
            cov + self.tikhonov_lambda*self.I
        )[0]))


    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def get_config(self):
        """Return necessary data to serialize the layer"""
        # Call parent's config
        config = super().get_config()
        # Update config with custom attributes
        config.update({
            # Base attributes
            'Dh': self.Dh,
            'Dout': self.Dout,
            'activation': self.activation,
            'activation2': self.activation2,
            'regularize': self.regularize,
            'beta': self.beta,
            'spectral_strategy': self.spectral_strategy,
            'W1_initializer': tf.keras.initializers.serialize(
                self.W1_initializer
            ),
            'W1_regularizer': tf.keras.regularizers.serialize(
                self.W1_regularizer
            ),
            'W1_constraint': tf.keras.constraints.serialize(
                self.W1_constraint
            ),
            'W2_initializer': tf.keras.initializers.serialize(
                self.W2_initializer
            ),
            'W2_regularizer': tf.keras.regularizers.serialize(
                self.W2_regularizer
            ),
            'W2_constraint': tf.keras.constraints.serialize(
                self.W2_constraint
            )
        })
        # Return updated config
        return config

    @classmethod
    def from_config(cls, config):
        """Use given config data to deserialize the layer"""
        # Instantiate layer
        hl = cls(**config)
        # Return deserialized layer
        return hl
