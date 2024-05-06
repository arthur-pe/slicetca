import torch
import numpy as np
from typing import Iterable
import warnings


def block_mask(dimensions: Iterable[int],
               train_blocks_dimensions: Iterable[int],
               test_blocks_dimensions: Iterable[int],
               number_blocks: int = None,
               fraction_test: float = 0.1,
               exact:bool = True,
               device:str = 'cpu'):
    """
    Builds train and test masks.
    The train mask has block of entries masked.
    The test mask has the opposite entries masked, plus the boundaries of the blocks.

    :param dimensions: Dimensions of the mask.
    :param train_blocks_dimensions: Dimensions of the blocks discarded for training will be 2*train_blocks_dimensions+1
    :param test_blocks_dimensions: Dimensions of the blocks retained for testing will be 2*test_blocks_dimensions+1
    :param number_blocks: The number of blocks. Deprecated, use fraction_test.
    :param fraction_test: The maximum fraction of entries in the test_mask
    :param exact:   If exact then the number of blocks will be number_blocks (slower).
                    If not exact, the number of blocks will be on average number_blocks (faster).
    :param device: torch device (e.g. 'cuda' or 'cpu').
    :return: train_mask, test_mask
    """

    valence = len(dimensions)

    flattened_max_dim = np.prod(dimensions)

    if not np.prod((np.array(train_blocks_dimensions)-np.array(test_blocks_dimensions))>=0):
        raise Exception('For all i it should be that train_blocks_dimensions[i]>=test_blocks_dimensions[i].')

    if number_blocks is None:
        number_blocks = int(fraction_test * flattened_max_dim / np.prod(2*np.array(test_blocks_dimensions)+1))
    else:
        warnings.warn('The parameter number_blocks is deprecated, use fraction_test instead.', DeprecationWarning)

    if exact:
        start = torch.zeros(flattened_max_dim, device=device)
        start[:number_blocks] = 1
        start = start[torch.randperm(flattened_max_dim, device=device)]
        start = start.reshape(dimensions)
    else:
        start = (torch.rand(tuple(dimensions), device=device) < fraction_test).long()

    start_index = start.nonzero()
    number_blocks = len(start_index)

    # Build outer-blocks mask
    a = [[slice(torch.clip(start_index[j][i]-train_blocks_dimensions[i],min=0, max=dimensions[i]),
                torch.clip(start_index[j][i]+train_blocks_dimensions[i]+1,min=0, max=dimensions[i]))
          for i in range(valence)] for j in range(number_blocks)]

    train_mask = torch.full(dimensions, True, device=device)

    for j in a: train_mask[j] = 0

    # Build inner-blocks tensor
    a = [[slice(torch.clip(start_index[j][i]-test_blocks_dimensions[i],min=0, max=dimensions[i]),
                torch.clip(start_index[j][i]+test_blocks_dimensions[i]+1,min=0, max=dimensions[i]))
                for i in range(valence)] for j in range(number_blocks)]

    test_mask = torch.full(dimensions, False, device=device)

    for j in a: test_mask[j] = 1

    return train_mask, test_mask
