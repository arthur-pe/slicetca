from matplotlib import pyplot as plt
import numpy as np
from typing import Sequence, Union


def plot(model,
         components: Sequence[Sequence[np.ndarray]] = None,
         variables: Sequence[str] = ('trial', 'neuron', 'time'),
         colors: Union[Sequence[np.ndarray], Sequence[Sequence[float]]] = (None, None, None),
         factor_height: int = 2,
         aspect: str = 'auto',
         s: int = 10,
         cmap: str = None,
         dpi: int = 60):
    """
    Plots SliceTCA components. Plotting TCA or PartitionTCA components also works but is not optimized.

    :param model: SliceTCA, TCA or PartitionTCA instance.
    :param components: By default, components = model.get_components(numpy=True).
                       But you may pass pre-processed components (e.g. sorted neurons etc...).
    :param variables: The axes labels, in the same order as the dimensions of the tensor.
    :param colors: The colors of the variable (e.g. trial condition). Used only for 1-tensor factors.
                   None or 1-d variable will default to plt.plot, 2-d (trials x RGBA) to scatter.
    :param factor_height: Height of the 1-tensor factors. Their length is 3.
    :param aspect: 'auto' will give a square-looking slice, 'equal' will preserve the ratios.
    :param s: size of scatter dots (see colors parameter).
    :param cmap: matplotlib cmap for 2-tensor factors (slices). Defaults to inferno for positive else seismic.
    :param dpi: Figure dpi. Set lower if you have many components of a given type.
    :return: A list of axes which can be used for further customizing the plots.
             The list has shape the same shape as model.get_components. That is component_type x (slice/factor) x rank
    """

    components = model.get_components(numpy=True) if components is None else components
    partitions = model.partitions
    positive = model.positive
    ranks = model.ranks

    number_nonzero_components = np.sum(np.array(ranks)!=0)

    axes = [[[None for k in j] for j in i] for i in components]

    figure_size = max([sum([j.shape[0]*3 if len(j.shape)==3 else j.shape[0]*factor_height for j in i]) for i in components])

    fig = plt.figure(figsize=(number_nonzero_components*3, figure_size), dpi=dpi, constrained_layout=True)
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

                    if p:
                        temp = current_component
                        ax.imshow(temp, aspect=aspect, cmap=(cmap if cmap is not None else 'inferno'))
                    else:
                        min_max = np.quantile(np.abs(current_component),0.95).item()
                        ax.imshow(current_component, aspect=aspect, cmap=(cmap if cmap is not None else 'seismic'),
                                  vmin=-min_max, vmax=min_max)

                    # =========== Axes labels ===========
                    variable_x = variables[partitions[i][k][1]]
                    variable_y = variables[partitions[i][k][0]]
                    ax.set_xlabel(variable_x)
                    ax.set_ylabel(variable_y)

                # =========== Higher order factors can't be plotted ===========
                elif len(list(components[i][k].shape)) >= 4:
                    ax = fig.add_subplot(gs[row:row+factor_height, column])
                    row += factor_height
                    ax.text(0.5, 0.5, '3$\geq$ tensor', va='center', ha='center', color='black')
                    ax.axis('off')

                axes[i][k][j] = ax

        if ranks[i] != 0: column += 1

    return axes


if __name__=='__main__':

    from slicetca._core.decompositions import SliceTCA, TCA, PartitionTCA

    #m = SliceTCA((10,15,20),(1,5,1), positive=True)
    #m = TCA((10,11,12), 3)

    #plot(m, aspect='auto', colors=(None, np.random.rand(15,3), None))

    m = PartitionTCA((5,10,15,20,25), [[[0],[1,2],[3,4]],[[0],[1],[2,3,4]]], [2,3], initialization='normal')
    plot(m, variables=[str(i) for i in range(5)])


    plt.tight_layout()
    plt.show()
