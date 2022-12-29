import torch
from torch import nn
import numpy as np

class partition_tca(nn.Module):

    def __init__(self, dimensions, partitions, ranks, positive=False, initialization='uniform',
                 init_weight=1.0, init_bias=0.0, device='cpu'):

        super(partition_tca, self).__init__()

        components = [[[dimensions[k] for k in j] for j in i] for i in partitions]

        positive_function = torch.abs if positive else self.identity

        vectors = nn.ModuleList([])

        for i in range(len(ranks)):
            rank = ranks[i]
            dim = components[i]

            # k-tensors of the outer product
            match initialization:
                case 'normal':
                    v = [nn.Parameter(positive_function[i][j](torch.randn([rank]+j, device=device))) for j in dim]
                case 'uniform':
                    v = [nn.Parameter(positive_function[i][j](torch.rand([rank] + j, device=device)*init_weight + init_bias)) for j in dim]
                case 'uniform-positive':
                    v = [nn.Parameter(positive_function[i][j](torch.rand([rank] + j, device=device)*init_weight + init_bias + init_weight)) for j in dim]
                case _:
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
        self.valence = len(dimensions)
        self.entries = np.prod(dimensions)

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

    def construct_single_component(self, type, k):

        temp2 = [self.positive_function(self.vectors[type][q][k]) for q in range(len(self.components[type]))]
        outer = torch.einsum(self.einsums[type], temp2)
        outer = outer.permute(self.inverse_permutations[type])

        return outer

    def construct(self):

        temp = torch.zeros(self.dims).to(self.device)

        for i in range(len(self.partitions)):
            for j in range(self.subranks[i]):
                if j not in self.skipped_indices[i]:
                    temp += self.construct_single_component(i, j)

        return temp

    def loss(self, a, b):
        return self.mse(a, b)

    def mse(self, a, b):
        return ((a-b)**2).mean()

    def get_components(self, detach=False, numpy=False):

        temp = [[] for i in range(len(self.vectors))]

        for i in range(len(self.vectors)):
            for j in range(len(self.vectors[i])):
                temp_index = torch.tensor(self.get_inverted_index(len(self.vectors[i][j]), i))
                if len(temp_index) != 0:
                    if not numpy:
                        if not detach:
                            temp[i].append(self.positive_function(self.vectors[i][j][temp_index]).data)
                        else:
                            temp[i].append(self.positive_function(self.vectors[i][j][temp_index]).data.detach())
                    else:
                        temp[i].append(
                            self.positive_function(self.vectors[i][j][temp_index]).data.detach().cpu().numpy())

        return temp

    def set_components(self, components):  # bug if positive_function != abs?

        for i in range(len(self.vectors)):
            for j in range(len(self.vectors[i])):
                temp_index = torch.tensor(self.get_inverted_index(len(self.vectors[i][j]), i))
                if len(temp_index) != 0:
                    with torch.no_grad():
                        if isinstance(components[i][j][temp_index], torch.Tensor):
                            self.vectors[i][j][temp_index].copy_(components[i][j][temp_index].to(self.device))  # critical change
                        else:
                            self.vectors[i][j][temp_index] += torch.tensor(components[i][j][temp_index], device=self.device)
        self.zero_grad()

    def init_train(self, max_iter=1000, min_delta=0.0001, steps_delta=10):
        self.max_iter = max_iter
        self.min_delta = min_delta
        self.steps_delta = steps_delta

    def mask_tensor(self, T, mask):

        if mask is None:
            return T
        else:
            return T*mask



    def fit(self, X, batch_prop=0.8, max_iter=1000, mask=None, optimizer=None, verbose_train=False, verbose_test=True):

        if optimizer is None: optimizer = torch.optim.AdamW(self.parameters(), lr=10**-2, weight_decay=10**-2)


        for i in range(max_iter):

            X_hat = self.construct()

            if batch_prop != 1.0: temp_mask = torch.rand(self.dimensions, device=self.device)<batch_prop
            if mask is not None: temp_mask = temp_mask & mask








class slice_tca(partition_tca):
    def __init__(self, dimensions, ranks, positive=False, initialization='uniform',
                 init_weight=1.0, init_bias=0.0, device='cpu'):

        valence = len(dimensions)
        partitions = [[[i], [j for j in range(valence) if j != i]] for i in range(valence)]

        super().__init__(dimensions=dimensions, ranks=ranks, partitions=partitions, positive=positive,
                         initialization=initialization, init_weight=init_weight, init_bias=init_bias, device=device)

class tca(partition_tca):
    def __init__(self, dimensions, rank, positive=False, initialization='uniform',
                 init_weight=1.0, init_bias=0.0, device='cpu'):

        if type(rank) is not tuple:
            rank = (rank,)

        valence = len(dimensions)
        partitions = [[[j] for j in range(valence)]]

        super().__init__(dimensions=dimensions, ranks=rank, partitions=partitions, positive=positive,
                         initialization=initialization, init_weight=init_weight, init_bias=init_bias, device=device)
