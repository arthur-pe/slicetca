from .helper_functions import squared_difference

import torch
from torch import nn
import numpy as np
import tqdm
from collections.abc import Iterable

from typing import Sequence, Union, Callable


class PartitionTCA(nn.Module):

    def __init__(self,
                 dimensions: Sequence[int],
                 partitions: Sequence[Sequence[Sequence[int]]],
                 ranks: Sequence[int],
                 positive: Union[bool, Sequence[Sequence[Callable]]] = False,
                 initialization: str = 'uniform',
                 init_weight: float = None,
                 init_bias: float = None,
                 device: str = 'cpu'):
        """
        Parent class for the sliceTCA and TCA decompositions.

        :param dimensions: Dimensions of the data to decompose.
        :param partitions: List of partitions of the legs of the tensor.
                        [[[0],[1]]] would be a matrix rank decomposition.
        :param ranks: Number of components of each partition.
        :param positive: If False does nothing.
                         If True constrains all components to be positive.
                         If list of list, the list of functions to apply to a given partition and component.
        :param initialization: Components initialization 'uniform'~U(-1,1), 'uniform-positive'~U(0,1), 'normal'~N(0,1).
        :param init_weight: Coefficient to multiply the initial component by.
        :param init_bias: Coefficient to add to the initial component.
        :param device: Torch device.
        """

        super(PartitionTCA, self).__init__()

        components = [[[dimensions[k] for k in j] for j in i] for i in partitions]

        if init_weight is None:
            if initialization == 'normal': init_weight = 1/np.sqrt(sum(ranks))
            if initialization == 'uniform-positive': init_weight = ((0.5 / sum(ranks)) ** (1 / max([len(p) for p in partitions])))*2
            if initialization == 'uniform': init_weight = 1/np.sqrt(sum(ranks))
        if init_bias is None: init_bias = 0.0

        if isinstance(positive, bool):
            if positive: positive_function = [[torch.abs for j in i] for i in partitions]
            else: positive_function = [[self.identity for j in i] for i in partitions]
        elif isinstance(positive, tuple) or isinstance(positive, list): positive_function = positive

        vectors = nn.ModuleList([])

        for i in range(len(ranks)):
            r = ranks[i]
            dim = components[i]

            # k-tensors of the outer product
            if initialization == 'normal':
                v = [nn.Parameter(positive_function[i][j](torch.randn([r]+d, device=device)*init_weight + init_bias)) for j, d in enumerate(dim)]
            elif initialization == 'uniform':
                v = [nn.Parameter(positive_function[i][j](2*(torch.rand([r] + d, device=device)-0.5)*init_weight + init_bias)) for j, d in enumerate(dim)]
            elif initialization == 'uniform-positive':
                v = [nn.Parameter(positive_function[i][j](torch.rand([r] + d, device=device)*init_weight + init_bias)) for j, d in enumerate(dim)]
            else:
                raise Exception('Undefined initialization, select one of : normal, uniform, uniform-positive')

            vectors.append(nn.ParameterList(v))

        self.vectors = vectors

        self.dimensions = dimensions
        self.partitions = partitions
        self.ranks = ranks
        self.positive = positive
        self.initialization = initialization
        self.init_weight = init_weight
        self.init_bias = init_bias
        self.device = device

        self.components = components
        self.positive_function = positive_function
        self.valence = len(dimensions)
        self.entries = np.prod(dimensions)

        self.losses = []

        self.inverse_permutations = []
        self.flattened_permutations = []
        for i in self.partitions:
            temp = []
            for j in i:
                for k in j:
                    temp.append(k)
            self.flattened_permutations.append(temp)
            self.inverse_permutations.append(torch.argsort(torch.tensor(temp)).tolist())

        self.set_einsums()

    def identity(self, x):
        return x

    def set_einsums(self):

        self.einsums = []
        for i in self.partitions:
            lhs = ''
            rhs = ''
            for j in range(len(i)):
                for k in i[j]:
                    lhs += chr(105 + k)
                    rhs += chr(105 + k)
                if j != len(i) - 1:
                    lhs += ','
            self.einsums.append(lhs + '->' + rhs)

    def construct_single_component(self, partition: int, k: int):
        """
        Constructs the kth term of the given partition.

        :param partition: Type of the partition
        :param k: Number of the component
        :return: Tensor of shape self.dimensions
        """

        temp = [self.positive_function[partition][q](self.vectors[partition][q][k]) for q in range(len(self.components[partition]))]
        outer = torch.einsum(self.einsums[partition], temp)
        outer = outer.permute(self.inverse_permutations[partition])

        return outer

    def construct_single_partition(self, partition: int):
        """
        Constructs the sum of the terms of a given type of partition.

        :param partition: Type of the partition
        :return: Tensor of shape self.dimensions
        """

        temp = torch.zeros(self.dimensions).to(self.device)
        for j in range(self.ranks[partition]):
            temp += self.construct_single_component(partition, j)

        return temp

    def construct(self):
        """
        Constructs the full tensor.
        :return: Tensor of shape self.dimensions
        """

        temp = torch.zeros(self.dimensions).to(self.device)

        for i in range(len(self.partitions)):
            for j in range(self.ranks[i]):
                temp += self.construct_single_component(i, j)

        return temp

    def get_components(self, detach=False, numpy=False):
        """
        Returns the components of the model.
        
        :param detach: Whether to detach the gradient.
        :param numpy: Whether to cast them to numpy arrays.
        :return: list of list of tensors.
        """

        temp = [[] for i in range(len(self.vectors))]

        for i in range(len(self.vectors)):
            for j in range(len(self.vectors[i])):
                if numpy:
                    temp[i].append( self.positive_function[i][j](self.vectors[i][j]).data.detach().cpu().numpy())
                else:
                    if not detach: temp[i].append(self.positive_function[i][j](self.vectors[i][j]).data.detach())
                    else: temp[i].append(self.positive_function[i][j](self.vectors[i][j]).data)

        return temp

    def set_components(self, components: Sequence[Sequence[torch.Tensor]]):  # bug if positive_function != abs
        """
        Set the model's components. 
        If the positive functions are abs or the identity model.set_components(model.get_components) 
        has no effect besides resetting the gradient.
        
        :param components: list of list tensors.
        """

        for i in range(len(self.vectors)):
            for j in range(len(self.vectors[i])):
                with torch.no_grad():
                    if isinstance(components[i][j], torch.Tensor):
                        self.vectors[i][j].copy_(components[i][j].to(self.device))
                    else:
                        self.vectors[i][j].copy_(torch.tensor(components[i][j], device=self.device))
        self.zero_grad()

    def fit(self,
            X: torch.Tensor,
            optimizer: torch.optim.Optimizer,
            loss_function: Callable = squared_difference,
            batch_prop: float = 0.2,
            max_iter: int = 10000,
            min_std: float = 10 ** -3,
            iter_std: int = 100,
            mask: torch.Tensor = None,
            verbose: bool = False,
            progress_bar: bool = True):
        """
        Fits the model to data.

        :param X: The data tensor.
        :param optimizer: A torch optimizer.
        :param loss_function: The final loss if torch.mean(loss_function(X, X_hat)). That is, loss_function: R^n -> R^n.
        :param batch_prop: Proportion of entries used to compute the gradient at every training iteration.
        :param max_iter: Maximum training iterations.
        :param min_std: Minimum std of the loss under which to return.
        :param iter_std: Number of iterations over which this std is computed.
        :param mask: Entries which are not used to compute the gradient at any training iteration.
        :param verbose: Whether to print the loss at every step.
        :param progress_bar: Whether to have a tqdm progress bar.
        """

        losses = []

        iterator = tqdm.tqdm(range(max_iter)) if progress_bar else range(max_iter)

        for iteration in iterator:

            X_hat = self.construct()

            loss_entries = loss_function(X, X_hat)

            total_loss = torch.mean(loss_entries)

            if batch_prop != 1.0: batch_mask = torch.rand(self.dimensions, device=self.device) < batch_prop

            if mask is None and batch_prop == 1.0:
                loss = total_loss
            else:
                if mask is None:
                    total_mask = batch_mask
                else:
                    if batch_prop == 1.0:
                        total_mask = mask
                    else:
                        total_mask = mask & batch_mask

                total_entries = torch.sum(total_mask)
                loss = torch.sum(loss_entries * total_mask) / total_entries

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss = total_loss.item()

            losses.append(total_loss)

            if verbose: print('Iteration:', iteration, 'Loss:', total_loss)
            if progress_bar: iterator.set_description('Loss: ' + str(total_loss) + ' ')

            if len(losses) > iter_std and np.array(losses[-iter_std:]).std() < min_std:
                if progress_bar: iterator.set_description('The model converged. Loss: ' + str(total_loss) + ' ')
                break

        self.losses += losses


class SliceTCA(PartitionTCA):
    def __init__(self, 
                 dimensions: Sequence[int], 
                 ranks: Sequence[int], 
                 positive: bool = False, 
                 initialization: str = 'uniform',
                 init_weight: float = None,
                 init_bias: float = None,
                 device: str = 'cpu'):
        """
        Main sliceTCA decomposition class.

        :param dimensions: Dimensions of the data to decompose.
        :param ranks: Number of components of each slice type.
        :param positive: If False does nothing.
                         If True constrains all components to be positive.
                         If list of list, the list of functions to apply to a given partition and component.
        :param initialization: Components initialization 'uniform'~U(-1,1), 'uniform-positive'~U(0,1), 'normal'~N(0,1).
        :param init_weight: Coefficient to multiply the initial component by.
        :param init_bias: Coefficient to add to the initial component.
        :param device: Torch device.
        """

        valence = len(dimensions)
        partitions = [[[i], [j for j in range(valence) if j != i]] for i in range(valence)]

        super().__init__(dimensions=dimensions, ranks=ranks, partitions=partitions, positive=positive,
                         initialization=initialization, init_weight=init_weight, init_bias=init_bias, device=device)


class TCA(PartitionTCA):
    def __init__(self,
                 dimensions: Sequence[int],
                 rank: int,
                 positive: bool = False,
                 initialization: str = 'uniform',
                 init_weight: float = None,
                 init_bias: float = None,
                 device: str = 'cpu'):
        """
        Main TCA decomposition class.

        :param dimensions: Dimensions of the data to decompose.
        :param rank: Number of components.
        :param positive: If False does nothing.
                         If True constrains all components to be positive.
                         If list of list, the list of functions to apply to a given partition and component.
        :param initialization: Components initialization 'uniform'~U(-1,1), 'uniform-positive'~U(0,1), 'normal'~N(0,1).
        :param init_weight: Coefficient to multiply the initial component by.
        :param init_bias: Coefficient to add to the initial component.
        :param device: Torch device.
        """

        if not isinstance(rank, Iterable):
            rank = (rank,)

        valence = len(dimensions)
        partitions = [[[j] for j in range(valence)]]

        super().__init__(dimensions=dimensions, ranks=rank, partitions=partitions, positive=positive,
                         initialization=initialization, init_weight=init_weight, init_bias=init_bias, device=device)
