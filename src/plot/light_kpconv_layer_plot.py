# ---   IMPORTS   --- #
# ------------------- #
from src.plot.mpl_plot import MplPlot
from src.plot.plot import PlotException
from src.plot.plot_utils import PlotUtils
from src.plot.kpconv_layer_plot import KPConvLayerPlot
import src.main.main_logger as LOGGING
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import os


# ---   CLASS   --- #
# ----------------- #
class LightKPConvLayerPlot(MplPlot):
    """
    :author: Alberto M. Esmoris Pena

    Class to plot the insights of a light KPConv layer.

    See :class:`.LightKPConvLayerPlot` and :class:`.LightKPConvLayer`.

    :ivar Q: The matrix representing the kernel's structure.
    :vartype Q: :class:`np.ndarray`
    :ivar W: The matrix representing the kernel's weights.
    :vartype W: :class:`np.ndarray`
    :ivar A: The matrix representing the kernel's scale factors.
    :vartype A: :class:`np.ndarray`
    :ivar Wpast: The matrices representing the kernel's weights at a previous
        state.
    :vartype Wpast: :class:`np.ndarray`
    :ivar sigma: The influence distance of the kernel.
    :vartype sigma: float
    :ivar name: The name of the layer containing the kernel.
    :vartype name: str
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of LightKPConvLayerPlot.

        :param kwargs: The key-word arguments defining the plot's attributes.
        :type kwargs: dict
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Initialize attributes of LightKPConvLayerPlot
        self.Q = kwargs.get('Q', None)
        self.W = kwargs.get('W', None)
        self.A = kwargs.get('A', None)
        self.Wpast = kwargs.get('Wpast', None)
        self.Apast = kwargs.get('Apast', None)
        self.sigma = kwargs.get('sigma', None)
        self.name = kwargs.get('name', None)
        # Validate
        if (
            self.Q is None or len(self.Q) < 1
        ) and (
            self.W is None or len(self.W) < 1
        ) and (
            self.A is None or len(self.A) < 1
        ):
            raise PlotException(
                'LightKPConvLayerPlot MUST receive at least the kernel '
                'structure (Q), the kernel weights (W), or the scale '
                'factors (A).'
            )
        if self.Q is not None and len(self.Q) > 0 and self.sigma is None:
            raise PlotException(
                'LightKPConvLayerPlot MUST receive the influence distance '
                '(sigma) if the structure space is given.'
            )

    # ---   PLOT METHODS   --- #
    # ------------------------ #
    def plot(self, **kwargs):
        """
        Plot the structure space, the matrix of weights, and the scale factors
        representing the kernel of the light KPConv layer.

        See :meth:`plot.Plot.plot`.
        """
        # Start time measurement
        start = time.perf_counter()
        # Do the plots
        if self.Q is not None:
            KPConvLayerPlot(
                Q=self.Q,
                sigma=self.sigma,
                name=self.name,
                path=self.path,
                show=self.show
            ).plot(**kwargs)
        if self.W is not None:
            self.plot_kernel_weights(
                self.W,
                'Winit' if self.Wpast is None else 'Wend',
                f'"{self.name}" weights',
                **kwargs
            )
            if self.Wpast is not None:
                self.plot_kernel_weights(
                    self.W-self.Wpast,
                    'Wdiff',
                    f'"{self.name}" weights diff.',
                    **kwargs
                )
        if self.A is not None:
            self.plot_scale_factors(
                self.A,
                'Ainit' if self.Apast is None else 'Aend',
                f'"{self.name}" scale factors',
                **kwargs
            )
            if self.Apast is not None:
                self.plot_scale_factors(
                    self.A-self.Apast,
                    'Adiff',
                    f'"{self.name}" scale factors diff.',
                    **kwargs
                )
        # End time measurement
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'LightKPConvLayerPlot generated and wrote figutes to '
            f'"{self.path}" in {end-start:.3f} seconds.'
        )

    def plot_kernel_weights(self, W, plot_name, plot_title, **kwargs):
        """
        Plot the kernel's weights.

        :param W: The matrix representing the weights of the kernel.
        :param plot_name: The name of the plot.
        :param plot_title: The title of the plot.
        :param kwargs: The key-word arguments.
        :return: Nothing, but the plot is written to a file.
        """
        # Plot weights as matrices
        self._plot_kernel_weights(
            W,
            plot_name+'_mat',
            plot_title,
            KPConvLayerPlot.plot_kernel_weights_as_matrix,
            **kwargs
        )
        # Plot weights as histograms
        self._plot_kernel_weights(
            W,
            plot_name+'_hist',
            plot_title,
            KPConvLayerPlot.plot_kernel_weights_as_historgram,
            **kwargs
        )

    def _plot_kernel_weights(
        self, W, plot_name, plot_title, plot_function, **kwargs
    ):
        """
        Assist the :meth:`.LightKPConvLayerPlot.plot_kernel_weights` function.
        """
        # Prepare figure
        fig = plt.figure(figsize=(12, 7))
        fig.suptitle(plot_title)
        # Plot weights matrix
        ax = fig.add_subplot(1, 1, 1)
        plot_function(fig, ax, W)
        # Post-process figure
        fig.tight_layout()
        # Make plot effective
        path = self.path
        self.path = os.path.join(path, f'{plot_name}.svg')
        self.save_show_and_clear(out_prefix=kwargs.get('out_prefix', None))
        self.path = path

    def plot_scale_factors(self, A, plot_name, plot_title, **kwargs):
        """
        Plot the scale factors.

        :param A: The matrix representing the scale factors.
        :type A: :class:`np.ndarray`
        :param plot_name: The name of the plot.
        :param plot_title: The title of the plot.
        :param kwargs: The key-word arguments.
        :return: Nothing, but the plot is written to a file.
        """
        # Prepare figure
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle(plot_title)
        # Plot scale factors matrix
        ax = fig.add_subplot(1, 2, 1)
        mat = ax.matshow(A, origin='lower', cmap='turbo')
        ax.xaxis.tick_bottom()
        fig.colorbar(mat, ax=ax)
        # Plot scale factors histogram
        ax = fig.add_subplot(1, 2, 2)
        ax.hist(A.flatten())
        # Post-process figure
        fig.tight_layout()
        # Make plot effective
        path = self.path
        self.path = os.path.join(path, plot_name)
        self.save_show_and_clear(out_prefix=kwargs.get('out_prefix', None))
        self.path = path
