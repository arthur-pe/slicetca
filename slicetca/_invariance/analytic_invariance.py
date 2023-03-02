import torch


def svd_basis(model, **kwargs):
    """
    Sets the vectors of each slice type to an orthonormal basis.

    :param model: SliceTCA model
    :param kwargs: ignored
    :return: model with new components.
    """

    device = model.device
    ranks = model.ranks

    new_components = [[None, None] for i in range(len(ranks))]
    for i in range(len(ranks)):
        if ranks[i] != 0:
            constructed = model.construct_single_partition(i)
            flattened_constructed = constructed.permute([i]+[q for q in range(len(ranks)) if q != i])
            flattened_constructed = flattened_constructed.reshape(model.dimensions[i],-1).transpose(0,1)

            U, S, V = torch.linalg.svd(flattened_constructed.detach().cpu(), full_matrices=False)
            U, S, V = U[:,:ranks[i]], S[:ranks[i]], V[:ranks[i]]
            U, S, V = U.to(device), S.to(device), V.to(device)

            US = (U @ torch.diag(S))
            slice = US.transpose(0,1).reshape([ranks[i]]+[model.dimensions[q] for q in range(len(ranks)) if q != i])

            new_components[i][0] = V
            new_components[i][1] = slice
        else:
            new_components[i][0] = torch.zeros_like(model.vectors[i][0])
            new_components[i][1] = torch.zeros_like(model.vectors[i][1])

    model.set_components(new_components)

    return model
