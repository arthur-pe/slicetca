from .transformations import *
from .criteria import *

import torch
import copy
import tqdm
from typing import Callable, Sequence


def invariance(model : Sequence[Sequence[torch.Tensor]],
               objective_function: Callable = l2,
               transformation: object = None,
               learning_rate: float = 10**-2,
               max_iter: int = 10000,
               min_std: float = 10**-3,
               iter_std: int = 100,
               verbose: bool = False,
               progress_bar: bool = True):

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
        if progress_bar: iterator.set_description("Invariance loss: " + str(l.item())[:10] + '')

        optim.zero_grad()
        l.backward()
        optim.step()

        losses.append(l.item())

        if len(losses)>iter_std and np.array(losses[-100:]).std()<min_std: break

    return transformation(components)


if __name__=='__main__':

    from slicetca._core.decompositions import SliceTCA
    m = SliceTCA((100,200,80),(3,2,4), initialization='uniform', device='cuda')

    a = m.construct().detach()

    transfo = nn.Sequential(TransformationBetween(m), TransformationWithin(m))
    #transfo = TransformationWithin(m)

    c = invariance(m, l2, transformation=transfo, verbose=True, max_iter=1000, learning_rate=0.01)
    m.set_components(c)
    b = m.construct()
    print(torch.mean(torch.square(a-b)))