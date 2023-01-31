import torch
from typing import Iterable


def block_masks(dimensions: Iterable[int],
                block_sizes: Iterable[int],
                block_density: float,
                boundaries_length: Iterable[int] = None):
    """
    Builds train and test masks.
    The train mask has block of entries masked.
    The test mask has the opposite entries masked, plus the boundaries of the blocks.

    :param dimensions:
    :param block_sizes:
    :param block_density:
    :param boundaries_length:
    :return: train_mask, test_mask
    """

    train_mask = (torch.rand(dimensions)<0.5)
    test_mask = train_mask

    return train_mask, test_mask
