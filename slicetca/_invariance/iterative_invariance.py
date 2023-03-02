from .transformations import *
from .criteria import *
from .._core.decompositions import SliceTCA

import torch
import copy
import tqdm
from typing import Callable


def sgd_invariance(model: SliceTCA,
                   objective_function: Callable = l2,
                   transformation: object = None,
                   learning_rate: float = 10**-2,
                   max_iter: int = 10000,
                   min_std: float = 10**-3,
                   iter_std: int = 100,
                   verbose: bool = False,
                   progress_bar: bool = True,
                   **kwargs):
    """
    Enables optimizing the components w.r.t. some objective function while fixing the overall reconstructed tensor.

    :param model: SliceTCA model.
    :param objective_function: The objective to optimize.
    :param transformation:  transformations.TransformationBetween(model) or
                            transformations.TransformationWithin(model) or
                            nn.Sequential(TransformationWithin(model), TransformationBetween(model))
    :param learning_rate: Learning rate for the optimizer (default Adam).
    :param max_iter: Maximum number of iterations.
    :param min_std: Minimum std of the last iter_std iterations under which to assume the model has converged.
    :param iter_std: See min_std.
    :param verbose: Whether to print the loss.
    :param progress_bar: Whether to have a progress bar.
    :param kwargs: ignored.
    :return: model with the modified components.
    """

    if transformation is None: transformation = TransformationBetween(model)

    model.requires_grad_(False)

    optim = torch.optim.Adam(transformation.parameters(), lr=learning_rate)

    components = model.get_components(detach=True)

    losses = []

    iterator = tqdm.tqdm(range(max_iter)) if progress_bar else range(max_iter)

    for iteration in iterator:

        components_transformed = transformation(copy.deepcopy(components))

        components_transformed_constructed = construct_per_type(model, components_transformed)
        l = objective_function(components_transformed_constructed)

        if verbose: print('Iteration:', iteration, '\tloss:', l.item())
        if progress_bar: iterator.set_description("Invariance loss: " + str(l.item())[:10] + ' ')

        optim.zero_grad()
        l.backward()
        optim.step()

        losses.append(l.item())

        if len(losses)>iter_std and np.array(losses[-100:]).std()<min_std:
            iterator.set_description("The invariance converged. Invariance loss: " + str(l.item()) + ' ')
            break

    model.set_components(transformation(components))

    return model
