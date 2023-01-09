from slicetca._run.decompose import decompose

from multiprocessing import Pool
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor as Pool
import torch
import numpy as np

from typing import Sequence

def grid_search(data: torch.Tensor,
                max_ranks: Sequence[int],
                mask: torch.Tensor = None,
                min_ranks: Sequence[int] = None,
                sample_size: int = 1,
                processes_sample: int = 1,
                processes_grid: int = 1,
                seed: int = 7,
                **kwargs):
    """
    Performs a gridsearch over different number of components (ranks) to see which has the lowest cross-validated loss.

    :param data: Data tensor to decompose.
    :param max_ranks: Maximum number of components of each type.
    :param mask: Mask representing over which entries to compute the loss of the gridsearch. None is full tensor.
    :param min_ranks: Minimum number of components of each type.
    :param sample_size: Number of seeds to use for a given number of components.
    :param processes_sample: Number of processes (threads) to use for a given number of components across seeds.
    :param processes_grid: Number of processes (threads) to use over different number of components.
    :param seed: Numpy seed.
    :param kwargs: Same kwargs as decompose.
    :return: A (max_rank_1-min_rank_1, max_rank_2-min_rank_2, ..., sample_size) ndarray of losses masked entries.
    """

    np.random.seed(seed)

    mp.set_start_method('spawn')

    if min_ranks is None: min_ranks = [0 for i in max_ranks]
    max_ranks = [i+1 for i in max_ranks]
    rank_span = [max_ranks[i]-min_ranks[i] for i in range(len(max_ranks))]

    grid = get_grid_sample(min_ranks, max_ranks)
    grid = np.concatenate([grid, np.random.randint(10**2,10**6, grid.shape[0])[:,np.newaxis]], axis=-1)

    print('Grid size:', str(rank_span), '- sample:', sample_size,
          '- total_fit:', torch.tensor(grid).size()[0]*sample_size)

    dec = partial(decompose_mp_sample, data=data, mask=mask, sample_size=sample_size, processes_sample=processes_sample, **kwargs)

    with Pool(max_workers=processes_grid) as pool: out_grid = np.array(list(pool.map(dec, grid)), dtype=np.float32)

    loss_grid = out_grid[:,0]
    seed_grid = out_grid[:,1].astype(int)

    loss_grid = loss_grid.reshape(rank_span+[sample_size])
    seed_grid = seed_grid.reshape(rank_span+[sample_size])

    return loss_grid, seed_grid


def decompose_mp_sample(number_components_seed, data, mask, sample_size, processes_sample, **kwargs):

    number_components = number_components_seed[:-1]
    seed = number_components_seed[-1]

    np.random.seed(seed)

    print('Starting fitting components:', number_components)

    dec = partial(decompose_mp, data=data.clone(), mask=mask, **kwargs)

    sample = number_components[np.newaxis].repeat(sample_size, 0)
    seeds = np.random.randint(10**2,10**6, sample_size)

    sample = np.concatenate([sample, seeds[:,np.newaxis]], axis=-1)

    with Pool(max_workers=processes_sample) as pool: loss = np.array(list(pool.map(dec, sample)))

    return loss, seeds


def decompose_mp(number_components_seed, data, mask, *args, **kwargs):

    number_components, seed = number_components_seed[:-1], number_components_seed[-1]

    if (number_components == np.zeros_like(number_components)).all():
        data_hat = 0
    else:
        _, model = decompose(data, number_components, verbose=False, progress_bar=False, *args, seed=seed, **kwargs)
        data_hat = model.construct()

    if mask is None: loss = torch.mean((data-data_hat)**2).item()
    else: loss = torch.mean(((data-data_hat)*(1-mask))**2).item()

    return loss


def get_grid_sample(min_dims, max_dims):

    grid = np.meshgrid(*[np.array([i for i in range(min_dims[j],max_dims[j])]) for j in range(len(max_dims))])

    grid = np.stack(grid)

    return grid.reshape(grid.shape[0], -1).T


if __name__=='__main__':

    from slicetca._core.decompositions import SliceTCA, TCA

    torch.manual_seed(8)

    dim = (25,15,20)

    data = SliceTCA(dim, [1,2,3], device='cuda').construct().detach()

    components, model = grid_search(data, [2,0,3], learning_rate=10**-3, max_iter=10, sample_size=2)
