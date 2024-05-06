# SliceTCA

This library provides tools to perform [sliceTCA](https://www.biorxiv.org/content/10.1101/2023.03.01.530616v1).

___

<p align="center">
  <img width="700" src="https://raw.githubusercontent.com/arthur-pe/slicetca/fb51e536afad9bfab60b5fc1def764ad6af1983c/img/decomposition.svg">
</p>

## Installation 

```commandline
pip install slicetca
```

## Full documentation

The full documentation can be found [here](https://github.com/arthur-pe/slicetca/blob/master/documentation.md).

## Examples

### Quick example 

```python
import slicetca
import torch
from matplotlib import pyplot as plt

device = ('cuda' if torch.cuda.is_available() else 'cpu')

# your_data is a numpy array of shape (trials, neurons, time).
data = torch.tensor(your_data, dtype=torch.float, device=device)

# The tensor is decomposed into 2 trial-, 0 neuron- and 3 time-slicing components.
components, model = slicetca.decompose(data, (2,0,3))

# For a not positive decomposition, we apply uniqueness constraints
model = slicetca.invariance(model)

slicetca.plot(model)

plt.show()
```

### Notebook

See the [example notebook](https://github.com/arthur-pe/slicetca/blob/master/sliceTCA_notebook_1.ipynb) for an application of sliceTCA to publicly available neural data.

<a target="_blank" href="https://colab.research.google.com/github/arthur-pe/slicetca/blob/master/sliceTCA_notebook_1.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Reference

A. Pellegrino<sub>@</sub><sup>†</sup>, H. Stein<sup>†</sup>, N. A. Cayco-Gaijc<sub>@</sub>. (2024). Dimensionality reduction beyond neural subspaces with slice tensor component analysis. *Nature Neuroscience* [https://www.nature.com/articles/s41593-024-01626-2](https://www.nature.com/articles/s41593-024-01626-2).