from slicetca._core.decompositions import PartitionTCA, SliceTCA
from slicetca.invariance.analytic_invariance import svd_basis
from slicetca.invariance.iterative_invariance import TransformationWithin, TransformationBetween, sgd_invariance, l2
from slicetca._run.decompose import decompose
from slicetca._run.grid_search import grid_search

import torch
from torch import nn
import numpy as np


dimensions = (5, 6, 7, 8)
ranks = (1, 0, 2, 3)


def test_analytic_invariance():

    m = SliceTCA(dimensions, ranks, initialization='uniform', device=device)

    a = m.construct().detach()
    m = svd_basis(m)
    b = m.construct()

    assert torch.mean(torch.square(a - b)).item()<10**-6, 'Analytic invariance changes L1'

    print('test_analytic_invariance passed')


def test_iterative_invariance_within_between():

    m = SliceTCA(dimensions, ranks, initialization='uniform', device=device)

    a = m.construct().detach()

    transfo = nn.Sequential(TransformationBetween(m), TransformationWithin(m))

    m = sgd_invariance(m, l2, transformation=transfo, max_iter=3, learning_rate=0.01, progress_bar=progress_bar)
    b = m.construct()

    assert torch.mean(torch.square(a - b)).item()<10**-6, 'Iterative invariance L2, L3 changes L1'

    print('test_invariance_between passed')


def test_iterative_invariance_within():
    m = SliceTCA(dimensions, ranks, initialization='uniform', device=device)

    a = m.construct().detach()

    transfo = TransformationWithin(m)

    m = sgd_invariance(m, l2, transformation=transfo, max_iter=3, learning_rate=0.01, progress_bar=progress_bar)
    b = m.construct()

    assert torch.mean(torch.square(a - b)).item()<10**-6, 'Iterative invariance L3 changes L1'

    print('test_invariance_within passed')


def test_iterative_invariance_between():
    m = SliceTCA(dimensions, ranks, initialization='uniform', device=device)

    a = m.construct().detach()

    transfo = TransformationBetween(m)

    m = sgd_invariance(m, l2, transformation=transfo, max_iter=3, learning_rate=0.01, progress_bar=progress_bar)
    b = m.construct()

    assert torch.mean(torch.square(a - b)).item()<10**-6, 'Iterative invariance L2 changes L1'

    print('test_iterative_invariance_between passed')


def test_gridsearch():
    
    data = SliceTCA(dimensions, ranks, device=device).construct().detach()

    mask_train = torch.rand_like(data) < 0.5
    mask_test = mask_train & (torch.rand_like(data) < 0.5)

    loss_grid, seed_grid = grid_search(data, ranks, learning_rate=10 ** -3, max_iter=3, sample_size=2,
                                    processes_sample=2,
                                    processes_grid=2, mask_train=mask_train, mask_test=mask_test)

    print('test_gridsearch passed')


def test_decompose():

    a = SliceTCA(dimensions, ranks, device=device).construct().detach()
    
    c, m = decompose(a, ranks)
    
    b = m.construct()

    assert torch.mean(torch.square(a - b)).item()<10**-6, 'Decompose fails to decompose a reconstructed tensor'

    print('test_decompose passed')


def test_fit():
    t = PartitionTCA(dimensions[:3], [[[0], [1, 2]], [[2, 0], [1]]], [1, 2]).construct().detach()
    p = PartitionTCA(dimensions[:3], [[[0], [1, 2]], [[2, 0], [1]]], [1, 2])

    optim = torch.optim.Adam(p.parameters(), lr=10 ** -4)

    p.fit(t, optimizer=optim, min_std=10 ** -6, batch_prop=0.1, max_iter=3 * 10 ** 4, progress_bar=progress_bar)

    mse = torch.mean(torch.square(t - p.construct())).item()

    assert mse < 10 ** -3, 'Failed to decompose a reconstruction: ' + str(mse)

    print('test_fit passed')


if __name__ == '__main__':
    
    torch.manual_seed(7)
    
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    progress_bar = False
    
    test_analytic_invariance()
    test_iterative_invariance_within_between()
    test_iterative_invariance_between()
    test_iterative_invariance_within()
    test_gridsearch()
    test_fit()

    print('All tests passed.')
