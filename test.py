from slicetca import *
import torch

a = torch.randn(10,20,30)

_, _ = decompose(a, [2,3,4], max_iter=100)

if __name__=='__main__':
    _, _ = grid_search(a, [2,3,4], max_iter=10)