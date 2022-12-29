import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.utils.benchmark as benchmark

def perm(T, pop_size, num_samples):
    T_f = T.flatten()
    t = T_f[torch.randperm(pop_size, device=device)[:num_samples]]
    l = torch.mean(t**2)
    l.backward()

def choice(T, pop_size, num_samples):
    t =  T[torch.rand(list(T.shape), device=device)<num_samples/pop_size]
    l = torch.mean(t**2)
    l.backward()

device = 'cuda'

idx_fns = [perm, choice]

pop_size = np.logspace(1, 2.5, 10, dtype=int)

d = []
for n in pop_size:

    T = torch.randn([n, n, n], device=device, requires_grad=True)

    print(n, end=', ')

    d_temp = []

    for fn in idx_fns:
        pop_size = np.prod(list(T.shape))
        num_samples = int(0.5 * pop_size)

        t0 = benchmark.Timer(
            stmt=f"{fn.__name__}(T, n_p, num_samples)",
            setup=f"from __main__ import {fn.__name__}",
            globals={'T':T, 'n_p': pop_size, 'num_samples': num_samples}
        )

        d_temp.append(t0.timeit(11).raw_times[0])

    d.append(d_temp)

d = np.array(d)

print()

for i in d:
    print(i)

plt.plot(d[:,0],label= 'perm')
plt.plot(d[:,1], label='choice')
plt.legend()
plt.show()
