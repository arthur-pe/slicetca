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

model = slicetca.invariance(model)

slicetca.plot(model)

plt.show()
```

### Notebook

See the [example notebook]() for an application of sliceTCA to publicly available neural data.

## Reference

A. Pellegrino<sub>@</sub><sup>†</sup>, H. Stein<sup>†</sup>, N. A. Cayco-Gaijc<sub>@</sub>. (2023). Disentangling Mixed Classes of Covariability in Large-Scale Neural Data. [https://www.biorxiv.org/content/10.1101/2023.03.01.530616v1](https://www.biorxiv.org/content/10.1101/2023.03.01.530616v1).