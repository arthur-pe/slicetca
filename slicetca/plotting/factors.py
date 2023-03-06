from matplotlib import pyplot as plt
import numpy as np
from typing import Sequence, Union


def plot(model,
         components: Sequence[Sequence[np.ndarray]] = None,
         variables: Sequence[str] = ('trial', 'neuron', 'time'),
         colors: Union[Sequence[np.ndarray], Sequence[Sequence[float]]] = (None, None, None),
         sorting_indices: Sequence[np.ndarray] = (None, None, None),
         ticks: Sequence[np.ndarray] = (None, None, None),
         tick_labels: Sequence[np.ndarray] = (None, None, None),
         quantile: float = 0.95,
         factor_height: int = 2,
         aspect: str = 'auto',
         s: int = 10,
         cmap: str = None,
         tight_layout: bool = True,
         dpi: int = 60):
    """
    Plots SliceTCA components. Plotting TCA or PartitionTCA components also works but is not optimized.

    :param model: SliceTCA, TCA or PartitionTCA instance.
    :param components: By default, components = model.get_components(numpy=True).
                       But you may pass pre-processed components (e.g. sorted neurons etc...).
    :param variables: The axes labels, in the same order as the dimensions of the tensor.
    :param colors: The colors of the variable (e.g. trial condition). Used only for 1-tensor factors.
                   None or 1-d variable will default to plt.plot, 2-d (trials x RGBA) to scatter.
                   Note that to generate RGBA colors from integer trial condition you may call:
                        colors = matplotlib.colormaps['hsv'](condition/np.max(condition))
    :param sorting_indices: Sort (e.g. trials) according to indices.
    :param ticks: Can be used instead of the 0,1, ... default indexing.
    :param tick_labels: Requires ticks
    :param quantile: Quantile of imshow cmap.
    :param factor_height: Height of the 1-tensor factors. Their length is 3.
    :param aspect: 'auto' will give a square-looking slice, 'equal' will preserve the ratios.
    :param s: size of scatter dots (see colors parameter).
    :param cmap: matplotlib cmap for 2-tensor factors (slices). Defaults to inferno for positive else seismic.
    :param tight_layout: To call plt.tight_layout(). Note that constrained_layout does not work well with
    :param dpi: Figure dpi. Set lower if you have many components of a given type.
    :return: A list of axes which can be used for further customizing the plots.
             The list has shape the same shape as model.get_components. That is component_type x (slice/factor) x rank
    """

    components = model.get_components(numpy=True) if components is None else components
    partitions = model.partitions
    positive = model.positive
    ranks = model.ranks

    # Pad the variables in case fewer than needed variables are provided
    variables = list(variables)+['variable '+str(i+1) for i in range(len(variables), max(len(variables), len(ranks)))]

    number_nonzero_components = np.sum(np.array(ranks) != 0)

    axes = [[[None for k in j] for j in i] for i in components]

    figure_size = max([sum([j.shape[0]*3 if len(j.shape) == 3 else j.shape[0]*factor_height for j in i]) for i in components])

    fig = plt.figure(figsize=(number_nonzero_components*3, figure_size), dpi=dpi)
    gs = fig.add_gridspec(figure_size, number_nonzero_components)

    column = 0
    for i in range(len(ranks)):
        row = 0
        for j in range(ranks[i]):
            for k in range(len(components[i])):
                current_component = components[i][k][j]

                # =========== Plots 1-tensor factors ===========
                if len(list(components[i][k].shape)) == 2:
                    ax = fig.add_subplot(gs[row:row+factor_height, column])
                    row += factor_height

                    leg = partitions[i][k][0]

                    if sorting_indices[leg] is not None:
                        current_component = current_component[sorting_indices[leg]]

                    if isinstance(colors[leg], np.ndarray) and len(colors[leg].shape) == 2:
                        ax.scatter(np.arange(len(current_component)), current_component, color=colors[leg], s=s)
                    else:
                        ax.plot(np.arange(len(current_component)), current_component,
                                color=(0.0, 0.0, 0.0) if colors[leg] is None else colors[leg])

                    ax.set_xlabel(variables[leg])

                # =========== Plots 2-tensor factors (slices) ===========
                elif len(list(components[i][k].shape)) == 3:
                    ax = fig.add_subplot(gs[row:row+3, column])
                    row += 3
                    ax.set_aspect(aspect)

                    p = (positive if isinstance(positive, bool) else positive[i][k])

                    if sorting_indices[partitions[i][k][0]] is not None:
                        current_component = current_component[sorting_indices[partitions[i][k][0]]]
                    if sorting_indices[partitions[i][k][1]] is not None:
                        current_component = current_component.T[sorting_indices[partitions[i][k][1]]].T

                    if p:
                        ax.imshow(current_component, aspect=aspect, cmap=(cmap if cmap is not None else 'inferno'),
                                  vmin=np.quantile(current_component,1-quantile),
                                  vmax=np.quantile(current_component,quantile))
                    else:
                        min_max = np.quantile(np.abs(current_component),quantile)
                        ax.imshow(current_component, aspect=aspect, cmap=(cmap if cmap is not None else 'seismic'),
                                  vmin=-min_max, vmax=min_max)

                    # =========== Axes labels ===========
                    variable_x = variables[partitions[i][k][1]]
                    variable_y = variables[partitions[i][k][0]]
                    ax.set_xlabel(variable_x)
                    ax.set_ylabel(variable_y)

                    if ticks[partitions[i][k][0]] is not None:
                        ax.set_yticks(ticks[partitions[i][k][0]], tick_labels[partitions[i][k][0]])
                    if ticks[partitions[i][k][1]] is not None:
                        ax.set_xticks(ticks[partitions[i][k][1]], tick_labels[partitions[i][k][1]])

                # =========== Higher order factors can't be plotted ===========
                elif len(list(components[i][k].shape)) >= 4:
                    ax = fig.add_subplot(gs[row:row+factor_height, column])
                    row += factor_height
                    ax.text(0.5, 0.5, '3$\geq$ tensor', va='center', ha='center', color='black')
                    ax.axis('off')

                # =========== Store axes ===========
                axes[i][k][j] = ax

        if ranks[i] != 0: column += 1

    if tight_layout: fig.tight_layout()

    return axes


if __name__=='__main__':

    from slicetca.core.decompositions import SliceTCA, TCA, PartitionTCA
    import matplotlib

    m = SliceTCA((10,15,20),(1,3,1), positive=False)
    #m = TCA((10,11,12), 3)

    print(np.random.randn(10,15)[np.random.permutation(10)].shape)

    trial_color = matplotlib.colormaps['hsv'](np.random.rand(10))
    axes = plot(m, aspect='auto', colors=(trial_color, None, None), dpi=60,
                ticks=(np.arange(10), np.arange(15), None),
                tick_labels=([chr(65+i) for i in range(10)], [chr(65+i) for i in range(15)], None),
                sorting_indices=(np.random.permutation(10), np.random.permutation(15), None))

    #m = PartitionTCA((5,10,15,20,25), [[[0],[1,2],[3,4]],[[0],[1],[2,3,4]]], [2,3], initialization='normal')
    #plot(m, variables=[str(i) for i in range(5)])

    #axes[1][0][1].axvline(5, color='grey', linestyle='--')

    plt.show()
