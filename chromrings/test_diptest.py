import numpy as np
import diptest

import matplotlib.pyplot as plt

# generate some bimodal random draws
N = 1000
hN = N // 2

dip_stats = []
dip_pvals = []

max_space = 5
sample_n = 20
sample_distributions = [0]*sample_n
multiplier = 10
n = sample_n*multiplier
sampling_step = n // sample_n
sampling_i = 0
space = np.linspace(0, max_space, n)
for i, m in enumerate(space):
    x = np.empty(N, dtype=np.float64)
    x[:hN] = np.random.normal(-m, 1.0, hN)*np.random.randint(1, 5)
    x[hN:] = np.random.normal(m, 1.0, hN)*np.random.randint(1, 5)

    # both the dip statistic and p-value
    dip, pval = diptest.diptest(x)

    dip_stats.append(dip)
    dip_pvals.append(pval)

    if (i == 0 or i%sampling_step == 0) and sampling_i<sample_n:
        sample_distributions[sampling_i] = (x, dip, pval)
        sampling_i += 1

print(f'dip stat = {dip}, p-value = {pval}')

sample_fig, sample_ax = plt.subplots(4,5)
for axes, (x, dip, pval) in zip(sample_ax.flatten(), sample_distributions):
    axes.hist(x, bins=20)
    axes.set_title(f'dip-stat = {dip:.3f}, p-value = {pval:.4f}')
    axes.tick_params(bottom=False, labelbottom=False)

sample_fig.subplots_adjust(
    left=0.05, bottom=0.04, top=0.95, right=0.95
)


fig, ax = plt.subplots(1,2)

ax[0].scatter(space, dip_stats)
ax[0].set_xlabel('Multimodality')
ax[0].set_ylabel('dip-test statstic')
ax[0].tick_params(bottom=False, labelbottom=False)

ax[1].scatter(space, dip_pvals)
ax[1].set_xlabel('Multimodality')
ax[1].set_ylabel('dip-test p-value')
ax[1].tick_params(bottom=False, labelbottom=False)

plt.show()