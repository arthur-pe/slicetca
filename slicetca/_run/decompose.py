import torch
from slicetca._core.decompositions import SliceTCA, TCA

def decompose(data, number_components,
              positive=False,
              initialization='uniform',
              optimizer=None,
              learning_rate = 10**-2,
              batch_prop=0.2,
              max_iter=10000,
              min_std=10**-3,
              iter_std=100,
              mask=None,
              verbose=True,
              seed=7):

    torch.manual_seed(seed)

    dimensions = list(data.shape)

    if isinstance(number_components, int): decomposition = TCA
    else: decomposition = SliceTCA

    model = decomposition(dimensions, number_components, positive, initialization, device=data.device)

    if optimizer is None: optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=10**-1)#

    model.fit(data, optimizer, batch_prop, max_iter, min_std, iter_std, mask, verbose)

    return model.get_components(numpy=True), model

if __name__=='__main__':

    torch.manual_seed(8)

    dim = (250,150,200)

    data = SliceTCA(dim, [1,2,3], device='cuda').construct().detach()

    components, model = decompose(data, [0,0,1], learning_rate=10**-3, max_iter=10)

    components, model = decompose(data, [1,2,3], learning_rate=10**-3, max_iter=10)

    components, model = decompose(data, [1,2,3], learning_rate=10**-3, max_iter=10)
