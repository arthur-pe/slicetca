from slicetca._invariance.iterative_invariance import sgd_invariance
from slicetca._invariance.analytic_invariance import svd_basis
from slicetca._invariance.criteria import *
from slicetca._core.decompositions import SliceTCA

dict_L2_invariance_objectives = {'regularization': l2}
dict_L3_invariance_functions = {'svd': svd_basis}

def invariance(model: SliceTCA,
               L2: str = 'regularization',
               L3: str = 'svd',
               **kwargs):
    """
    High level function for invariance optimization.
    Note: modifies inplace, deepcopy your model if you want a copy of the not invariance-optimized components.

    :param model: A sliceTCA model.
    :param L2: String, currently only supports 'regularization', you may add additional objectives.
    :param L3: String, currently only supports 'svd'.
    :param kwargs: Key-word arguments to be passed to L2 and L3 optimization functions. See iterative_function.py
    :return: model with modified components.
    """

    model = sgd_invariance(model, objective_function=dict_L2_invariance_objectives[L2], **kwargs)
    model = dict_L3_invariance_functions[L3](model, **kwargs)

    return model
