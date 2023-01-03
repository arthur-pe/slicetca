from slicetca._run.decompose import decompose

from multiprocessing import Pool
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor as Pool
import torch
import numpy as np


def decompose_mp_sample(number_components_seed, data, mask, sample_size, threads_sample, **kwargs):

    number_components = number_components_seed[:-1]
    seed = number_components_seed[-1]

    np.random.seed(seed)

    print('Starting fitting components:', number_components)

    dec = partial(decompose_mp, data=data.clone(), mask=mask, **kwargs)

    sample = number_components[np.newaxis].repeat(sample_size, 0)
    seeds = np.random.randint(10**2,10**6, sample_size)

    sample = np.concatenate([sample, seeds[:,np.newaxis]], axis=-1)

    with Pool(max_workers=threads_sample) as pool: loss = np.array(list(pool.map(dec, sample)))

    return loss, seeds


def decompose_mp(number_components_seed, data, mask, *args, **kwargs):

    number_components, seed = number_components_seed[:-1], number_components_seed[-1]

    if (number_components == np.zeros_like(number_components)).all():
        data_hat = 0
    else:
        _, model = decompose(data, number_components, verbose=True, *args, seed=seed, **kwargs)
        data_hat = model.construct()

    if mask is None: loss = torch.mean((data-data_hat)**2).item()
    else: loss = torch.mean(((data-data_hat)*(1-mask))**2).item()

    return loss


def get_grid_sample(min_dims, max_dims):

    grid = np.meshgrid(*[np.array([i for i in range(min_dims[j],max_dims[j])]) for j in range(len(max_dims))])

    grid = np.stack(grid)

    return grid.reshape(grid.shape[0], -1).T


def grid_search(data, max_ranks, mask=None, min_ranks=None, sample_size=1, threads_sample=1, threads_grid=1, seed=7, **kwargs):

    np.random.seed(seed)

    mp.set_start_method('spawn')

    if min_ranks is None: min_ranks = [0 for i in max_ranks]
    max_ranks = [i+1 for i in max_ranks]
    rank_span = [max_ranks[i]-min_ranks[i] for i in range(len(max_ranks))]

    grid = get_grid_sample(min_ranks, max_ranks)
    grid = np.concatenate([grid, np.random.randint(10**2,10**6, grid.shape[0])[:,np.newaxis]], axis=-1)

    print('Grid size:', str(rank_span), '- sample:', sample_size,
          '- total_fit:', torch.tensor(grid).size()[0]*sample_size)

    dec = partial(decompose_mp_sample, data=data, mask=mask, sample_size=sample_size, threads_sample=threads_sample, **kwargs)

    with Pool(max_workers=threads_grid) as pool: out_grid = np.array(list(pool.map(dec, grid)), dtype=np.float32)

    loss_grid = out_grid[:,0]
    seed_grid = out_grid[:,1].astype(int)

    loss_grid = loss_grid.reshape(rank_span+[sample_size])
    seed_grid = seed_grid.reshape(rank_span+[sample_size])

    return loss_grid, seed_grid

if __name__=='__main__':

    from slicetca._core.decompositions import SliceTCA, TCA

    torch.manual_seed(8)

    dim = (25,15,20)

    data = SliceTCA(dim, [1,2,3], device='cuda').construct().detach()

    components, model = grid_search(data, [2,0,3], learning_rate=10**-3, max_iter=10, sample_size=2)
