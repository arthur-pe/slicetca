from .transformations import *
from .criteria import *

import torch
import copy


def invariance(model, objective_function=l2, transformation=None,
           learning_rate=10**-2, max_iter=10000, min_std=10**-3, iter_std=100, verbose=False):

    if transformation is None: transformation = TransformationBetween(model)

    model.requires_grad_(False)

    optim = torch.optim.Adam(transformation.parameters(), lr=learning_rate)

    components = model.get_components(detach=True)

    losses = []

    for iteration in range(max_iter):

        components_transformed = transformation(copy.deepcopy(components))

        components_transformed_constructed = construct_per_type(model, components_transformed)
        l = objective_function(components_transformed_constructed)

        if verbose: print('Iteration:', iteration, '\tloss:', l.item())

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