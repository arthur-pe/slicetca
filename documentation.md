# Documentation

Here is a brief list of the functionalities provided by this repository. Additional information is provided in their docstring or by calling `help(function_name)`.

## High-level functions

To get quickly started the following high-level functions can be used. These can imported at once with `from slicetca import *`.

 * `decompose`  is the high-level function to decompose a data tensor.
 * `grid_search` is for determining the number of components.
 * `plot` allows plotting sliceTCA and TCA components.

We recommend having a look at our notebooks for further details.

## Low-level functions

For more specific use-cases, low-level functions might be preferred.

 * `_core.decompositions.SliceTCA`
   * `.fit(self, data, ...)` fits the components of a sliceTCA object to some data. 
   * `.set_components(self, components)` sets the model's components. 
   * `.get_components(self, detach=False, numpy=False)` returns the model's components. To backpropagate through the tensor, set detach=False.
 * `_invariance` 
   * `.analytic_invariance.svd_basis(model)` sets the vectors of each slice type to an orthonormal basis and sort them by variance explained.
   * `.sgd_invariance(model, objective_function, transformation, ...)` allows optimizing the components w.r.t. some objective function while fixing the overall tensor. 
 