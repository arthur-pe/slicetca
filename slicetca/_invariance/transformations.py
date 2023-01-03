from itertools import combinations
import torch
import torch.nn as nn
import numpy as np

from .helper import *

class TransformationBetween(nn.Module):

    def __init__(self, model):
        super(TransformationBetween, self).__init__()

        self.number_components = len(model.ranks)
        self.ranks = model.ranks
        self.partitions = model.partitions
        self.dims = model.dimensions
        self.device = model.device

        self.free_vectors_combinations = nn.ModuleList(
            [nn.ParameterList([nn.Parameter(torch.tensor(0.0, device=self.device)) for j in range(self.number_components)]) for i in
             range(self.number_components)])

        self.components = model.get_components()
        #self.model = copy.deepcopy(model)

        self.remaining_index_combinations = [[None for j in range(self.number_components)] for i in
                                             range(self.number_components)]
        self.remaining_dims_combinations = [[None for j in range(self.number_components)] for i in
                                            range(self.number_components)]

        for combination in combinations(list(range(self.number_components)), 2):
            if self.ranks[combination[0]] != 0 and self.ranks[combination[1]] != 0:
                temp = set(self.partitions[combination[0]][1])
                temp = temp.intersection(set(self.partitions[combination[1]][1]))
                self.remaining_index_combinations[combination[0]][combination[1]] = list(temp)

                self.remaining_dims_combinations[combination[0]][combination[1]] = [self.dims[i] for i in temp]
                remaining_dims = self.remaining_dims_combinations[combination[0]][combination[1]]

                free_vectors_dim = [self.ranks[combination[0]], self.ranks[combination[1]]] + remaining_dims
                free_vectors = nn.Parameter(torch.randn(free_vectors_dim, device=self.device))
                self.free_vectors_combinations[combination[0]][combination[1]] = free_vectors

    def forward(self, components):

        for combination in combinations(list(range(self.number_components)), 2):
            if self.ranks[combination[0]] != 0 and self.ranks[combination[1]] != 0:
                a_index = self.partitions[combination[0]][0][0]
                b_index = self.partitions[combination[1]][0][0]

                A_indexes = [b_index] + self.remaining_index_combinations[combination[0]][combination[1]]
                B_indexes = [a_index] + self.remaining_index_combinations[combination[0]][combination[1]]

                perm_B = [A_indexes.index(i) for i in self.partitions[combination[0]][1]]
                perm_A = [B_indexes.index(i) for i in self.partitions[combination[1]][1]]

                free_vectors = self.free_vectors_combinations[combination[0]][combination[1]]
                A = batch_outer(components[combination[0]][0], free_vectors)
                B = batch_outer(components[combination[1]][0], free_vectors.transpose(0, 1))

                A = A.sum(dim=0)
                B = B.sum(dim=0)

                A = A.transpose(0, 1)
                B = B.transpose(0, 1)

                A = A.permute([0] + [1 + i for i in perm_A])
                B = B.permute([0] + [1 + i for i in perm_B])

                components[combination[0]][1] = components[combination[0]][1] + B
                components[combination[1]][1] = components[combination[1]][1] - A

        return components


class TransformationWithin(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.ranks = model.ranks
        self.number_components = len(model.ranks)
        self.device = model.device

        self.free_gl = nn.ParameterList([nn.Parameter(torch.eye(i, device=self.device)+torch.randn((i,i),
                                                    device=self.device)/np.sqrt(3*i)) for i in self.ranks])

    def forward(self, components):

        for i in range(self.number_components):
            if self.ranks[i] != 0:
                components[i][0] = mm(self.free_gl[i].T, components[i][0])
                components[i][1] = mm(torch.linalg.inv(self.free_gl[i]), components[i][1])

        return components

