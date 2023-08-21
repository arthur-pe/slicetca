import torch


def squared_difference(x, x_hat):
    return (x - x_hat) ** 2


def poisson_log_likelihood(spikes, rates, spikes_factorial, activation=torch.nn.functional.softplus):

    likelihood = torch.exp(-activation(rates)) * torch.pow(activation(rates), spikes) / spikes_factorial

    return -torch.log(likelihood)
