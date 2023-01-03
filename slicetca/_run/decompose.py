from slicetca._core import SliceTCA, TCA

import torch
from typing import Union, Sequence
import numpy as np


def decompose(data: Union[torch.Tensor, np.array],
              number_components: Union[Sequence[int], int],
              positive: bool = False,
              initialization: str = 'uniform',
              learning_rate: float = 10**-2,
              batch_prop: float = 0.2,
              max_iter: int = 10000,
              min_std: float = 10**-3,
              iter_std: int = 100,
              mask: torch.Tensor = None,
              verbose: bool = True,
              seed: int = 7):
    """
    High-level function to decompose a data tensor into a SliceTCA or TCA decomposition.

    :param data: Torch tensor.
    :param number_components: If list or tuple number of sliceTCA components, else number of TCA components.
    :param positive: Whether to use a positive decomposition.
    :param initialization: Components initialization 'uniform'~U(-1,1), 'uniform-positive'~U(0,1), 'normal'~N(0,1).
    :param learning_rate: Learning rate of the optimizer.
    :param batch_prop: Proportion of entries used to compute the gradient at every training iteration.
    :param max_iter: Maximum training iterations.
    :param min_std: Minimum std of the loss under which to return.
    :param iter_std: Number of iterations over which this std is computed.
    :param mask: Entries which are not used to compute the gradient at any training iteration.
    :param verbose: Whether to print the loss at every step.
    :param seed: Torch seed.
    :return:  components: A list (over component types) of lists (over factors) of rank x component_shape tensors.
    :return: model: A SliceTCA or TCA model. It can be used to access the losses over training and much more.
    """

    torch.manual_seed(seed)

    dimensions = list(data.shape)

    if isinstance(number_components, int): decomposition = TCA
    else: decomposition = SliceTCA

    model = decomposition(dimensions, number_components, positive, initialization, device=data.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=10**-1)#

    model.fit(data, optimizer, batch_prop, max_iter, min_std, iter_std, mask, verbose)

    return model.get_components(numpy=True), model

if __name__=='__main__':

    torch.manual_seed(8)

    dim = (250,150,200)

    data = SliceTCA(dim, [1,2,3], device='cuda').construct().detach()

    components, model = decompose(data, [0,0,1], learning_rate=10**-3, max_iter=10)

    components, model = decompose(data, [1,2,3], learning_rate=10**-3, max_iter=10)

    components, model = decompose(data, [1,2,3], learning_rate=10**-3, max_iter=10)
