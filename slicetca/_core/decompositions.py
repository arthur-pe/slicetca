import torch
from torch import nn
import numpy as np


class PartitionTCA(nn.Module):

    def __init__(self, dimensions, partitions, ranks, positive=False, initialization='uniform',
                 init_weight=1.0, init_bias=0.0, device='cpu'):

        super(PartitionTCA, self).__init__()

        components = [[[dimensions[k] for k in j] for j in i] for i in partitions]

        if isinstance(positive, bool):
            if positive: positive_function = [[torch.abs for j in i] for i in partitions]
            else: positive_function = [[self.identity for j in i] for i in partitions]
        elif isinstance(positive, tuple) or isinstance(positive, list): positive_function = positive

        vectors = nn.ModuleList([])

        for i in range(len(ranks)):
            rank = ranks[i]
            dim = components[i]

            # k-tensors of the outer product
            match initialization:
                case 'normal':
                    v = [nn.Parameter(positive_function[i][j](torch.randn([rank]+d, device=device))) for j, d in enumerate(dim)]
                case 'uniform':
                    v = [nn.Parameter(positive_function[i][j](torch.rand([rank] + d, device=device)*init_weight + init_bias)) for j, d in enumerate(dim)]
                case 'uniform-positive':
                    v = [nn.Parameter(positive_function[i][j](torch.rand([rank] + d, device=device)*init_weight + init_bias + init_weight)) for j, d in enumerate(dim)]
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

    def construct_single_component(self, type, k):

        temp = [self.positive_function[type][q](self.vectors[type][q][k]) for q in range(len(self.components[type]))]
        outer = torch.einsum(self.einsums[type], temp)
        outer = outer.permute(self.inverse_permutations[type])

        return outer

    def construct_single_type(self, type):
        temp = torch.zeros(self.dimensions).to(self.device)
        for j in range(self.ranks[type]):
            temp += self.construct_single_component(type, j)

        return temp

    def construct(self):

        temp = torch.zeros(self.dimensions).to(self.device)

        for i in range(len(self.partitions)):
            for j in range(self.ranks[i]):
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
                if numpy:
                    temp[i].append( self.positive_function[i][j](self.vectors[i][j]).data.detach().cpu().numpy())
                else:
                    if not detach: temp[i].append(self.positive_function[i][j](self.vectors[i][j]).data.detach())
                    else: temp[i].append(self.positive_function[i][j](self.vectors[i][j]).data)

        return temp

    def set_components(self, components):  # bug if positive_function != abs

        for i in range(len(self.vectors)):
            for j in range(len(self.vectors[i])):
                with torch.no_grad():
                    if isinstance(components[i][j], torch.Tensor):
                        self.vectors[i][j].copy_(components[i][j].to(self.device))
                    else:
                        self.vectors[i][j].copy_(torch.tensor(components[i][j], device=self.device))
        self.zero_grad()

    def init_train(self, max_iter=1000, min_delta=0.0001, steps_delta=10):
        self.max_iter = max_iter
        self.min_delta = min_delta
        self.steps_delta = steps_delta

    def fit(self, X, optimizer, batch_prop=0.2, max_iter=10000, min_std=10**-3, iter_std=100, mask=None, verbose=True):

        losses = []

        for iteration in range(max_iter):

            X_hat = self.construct()

            dX = X-X_hat

            if mask is not None: dX = dX * mask

            with torch.no_grad(): total_loss = torch.mean(torch.square(dX)).item() #should be divided by prop masked entries

            if batch_prop != 1.0:
                dX = dX*(torch.rand(self.dimensions, device=self.device)<batch_prop)
                loss = torch.mean(torch.square(dX))/batch_prop
            else:
                loss = torch.mean(torch.square(dX))

            loss.backward()

            optimizer.step()

            losses.append(total_loss)

            if verbose: print('Iteration:', iteration, 'MSE loss:', total_loss)

            if len(losses)>iter_std and np.array(losses[-iter_std:]).std()<min_std: break

        self.losses += losses


class SliceTCA(PartitionTCA):
    def __init__(self, dimensions, ranks, positive=False, initialization='uniform',
                 init_weight=1.0, init_bias=0.0, device='cpu'):

        valence = len(dimensions)
        partitions = [[[i], [j for j in range(valence) if j != i]] for i in range(valence)]

        super().__init__(dimensions=dimensions, ranks=ranks, partitions=partitions, positive=positive,
                         initialization=initialization, init_weight=init_weight, init_bias=init_bias, device=device)


class TCA(PartitionTCA):
    def __init__(self, dimensions, rank, positive=False, initialization='uniform',
                 init_weight=1.0, init_bias=0.0, device='cpu'):

        if type(rank) is not tuple:
            rank = (rank,)

        valence = len(dimensions)
        partitions = [[[j] for j in range(valence)]]

        super().__init__(dimensions=dimensions, ranks=rank, partitions=partitions, positive=positive,
                         initialization=initialization, init_weight=init_weight, init_bias=init_bias, device=device)


if __name__=='__main__':

    t = PartitionTCA([11,12,13], [[[0],[1,2]], [[2,0], [1]]], [1,2]).construct().detach()
    t = torch.ones([11,12,13])+torch.randn([11,12,13])/2
    p = PartitionTCA([11,12,13], [[[0],[1,2]], [[2,0], [1]]], [1,0])

    optim = torch.optim.AdamW(p.parameters(), lr=10**-4, weight_decay=10**-3)
    #optim = torch.optim.SGD(p.parameters(), lr=10**-5)
    #optim = torch.optim.Adam(p.parameters(), lr=10**-4)
    p.fit(t, optimizer=optim, min_std=10**-6, batch_prop=0.1, max_iter=5*10**4)

    print(torch.mean(torch.square(t-p.construct())))

    print(p.vectors[0][0].std())
    print(p.vectors[0][1].std())
