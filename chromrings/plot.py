import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# import diptest

from chromrings import tables_path, core

import seaborn as sns

df_summary_path = os.path.join(tables_path, 'profiles_summary.csv')
df_summary = pd.read_csv(df_summary_path)


"""Summary boxplots"""
fig, ax = plt.subplots(1, 3, figsize=(18,8))
x = 'condition'

y = 'argmean'
sns.boxplot(x=x, y=y, data=df_summary, ax=ax[0])
ax[0].set_title('Mean of distance distribution')

y = 'argmax'
sns.boxplot(x=x, y=y, data=df_summary, ax=ax[1])
ax[1].set_title('Mode of distance distribution')

y = 'CV'
sns.boxplot(x=x, y=y, data=df_summary, ax=ax[2])
ax[2].set_title('CV of distance distribution')

plt.show()

"""Average profile starved vs fed"""
df_profiles_fed_path = os.path.join(tables_path, 'profiles_fed.csv')
df_profiles_fed = pd.read_csv(df_profiles_fed_path, index_col='dist_perc')
df_profiles_fed = df_profiles_fed / df_profiles_fed.max()
df_fed_average = df_profiles_fed.mean(axis=1).dropna()

df_profiles_starved_path = os.path.join(tables_path, 'profiles_starved.csv')
df_profiles_starved = pd.read_csv(df_profiles_starved_path, index_col='dist_perc')
df_profiles_starved = df_profiles_starved / df_profiles_starved.max()
df_starved_average = df_profiles_starved.mean(axis=1).dropna()

fig, ax = plt.subplots(1, 3)

colors = sns.color_palette(n_colors=2)

ax[0].plot(df_profiles_fed.index , df_profiles_fed.values, color=colors[0])
ax[0].plot(df_profiles_starved.index , df_profiles_starved.values, color=colors[1])
ax[0].set_title('Single cell profiles')

ax[1].plot(df_fed_average.index , df_fed_average.values, color=colors[0])
ax[1].plot(df_starved_average.index , df_starved_average.values, color=colors[1])

"""Find peaks"""
find_peaks_kwargs = {}
peaks_fed, props_fed = find_peaks(df_fed_average.values, **find_peaks_kwargs)
xx_peaks_fed = df_fed_average.index[peaks_fed]
yy_peaks_fed = df_fed_average.values[peaks_fed]

ax[1].scatter(xx_peaks_fed, yy_peaks_fed, s=80, facecolors='none', edgecolors=colors[0])

peaks_str, props_str = find_peaks(df_starved_average.values, **find_peaks_kwargs)
xx_peaks_str = df_starved_average.index[peaks_str]
yy_peaks_str = df_starved_average.values[peaks_str]

ax[1].scatter(xx_peaks_str, yy_peaks_str, s=80, facecolors='none', edgecolors=colors[1])
ax[1].set_title('Average of all profiles')

'''Fit gaussian model'''
model = core.PeaksModel()

# Fed fit
X_fed = df_fed_average.index.values
Y_fed = df_fed_average.values
init_guess_fed = model.get_init_guess(xx_peaks_fed, yy_peaks_fed, X_fed)
B_guess_fed = np.min(Y_fed)
result_fed = model.fit(X_fed, Y_fed, init_guess_fed, B_guess_fed)

X_pred_fed = np.linspace(X_fed.min(), X_fed.max(), 1000)
coeffs_pred_fed = result_fed.x[:-1]
B_pred_fed = result_fed.x[-1]
Y_pred_fed = model.model(X_pred_fed, coeffs_pred_fed, B_pred_fed)

ax[2].scatter(X_fed , Y_fed, color=colors[0])
ax[2].plot(X_pred_fed , Y_pred_fed, color=colors[0])

# Integrate analytically
n = 0
I_tot = 0
I_foregr_fed, I_tot_fed, Is_foregr_fed, Is_tot_fed = model.integrate(
    coeffs_pred_fed, B_pred_fed
)

# Integrate by summation
rolled_space = np.roll(X_pred_fed, -1)
rolled_space[-1] = X_pred_fed[-1] + (X_pred_fed[-1]-X_pred_fed[-2])
delta_space = rolled_space - X_pred_fed
I_foregr_sum_fed = (
    np.sum(Y_pred_fed*delta_space) 
    - B_pred_fed*(X_pred_fed.max()-X_pred_fed.min())
)

info_fed = ''
for i in range(len(xx_peaks_fed)):
    ratio = 100*Is_foregr_fed[i]/I_foregr_fed
    s = f'  - Peak nr. {i+1} area ratio percentage = {ratio:.2f}\n'
    info_fed = f'{info_fed}{s}'

print(
    '===========================================================\n'
    'Fed:\n'
    f'  - Foreground analytical integral = {I_foregr_fed:.3f}\n',
    f'  - Foreground discrete sum = {I_foregr_sum_fed:.3f}\n',
    f'  - Total analytical integral = {I_tot_fed:.3f}\n',
    f'{info_fed}'
    '===========================================================\n'
)

# Starved fit
X_str = df_starved_average.index.values
Y_str = df_starved_average.values
init_guess_str = model.get_init_guess(xx_peaks_str, yy_peaks_str, X_str)
B_guess_str = np.min(Y_str)
result_str = model.fit(X_str, Y_str, init_guess_str, B_guess_str)

X_pred_str = np.linspace(X_str.min(), X_str.max(), 1000)
coeffs_pred_str = result_str.x[:-1]
B_pred_str = result_str.x[-1]
Y_pred_str = model.model(X_pred_str, coeffs_pred_str, B_pred_str)

ax[2].scatter(X_str , Y_str, color=colors[1])
ax[2].plot(X_pred_str , Y_pred_str, color=colors[1])
ax[2].set_title('Gaussian fit')

# Integrate analytically
n = 0
I_tot = 0
I_foregr_str, I_tot_str, Is_foregr_str, Is_tot_str = model.integrate(
    coeffs_pred_str, B_pred_str
)

# Integrate by summation
rolled_space = np.roll(X_pred_str, -1)
rolled_space[-1] = X_pred_str[-1] + (X_pred_str[-1]-X_pred_str[-2])
delta_space = rolled_space - X_pred_str
I_foregr_sum_str = (
    np.sum(Y_pred_str*delta_space) 
    - B_pred_str*(X_pred_str.max()-X_pred_str.min())
)

info_str = ''
for i in range(len(xx_peaks_str)):
    ratio = 100*Is_foregr_str[i]/I_foregr_str
    s = f'  - Peak nr. {i+1} area ratio percentage = {ratio:.2f}%\n'
    info_str = f'{info_str}{s}'

print(
    '===========================================================\n'
    'Starved:\n'
    f'  - Foreground analytical integral = {I_foregr_str:.3f}\n',
    f'  - Foreground discrete sum = {I_foregr_sum_str:.3f}\n',
    f'  - Total analytical integral = {I_tot_str:.3f}\n',
    f'{info_str}'
    '===========================================================\n'
)

# ax[2].scatter(df_starved_average.index , df_starved_average.values, color=colors[1])

# # dip-test 
# probabilities = df_fed_average.values/df_fed_average.values.sum()
# x_fed = np.random.choice(df_fed_average.index, p=probabilities, size=10000)
# dip_fed, pval_fed = diptest.diptest(x_fed)

# probabilities = df_starved_average.values/df_starved_average.values.sum()
# x_str = np.random.choice(df_starved_average.index, p=probabilities, size=10000)
# dip_str, pval_str = diptest.diptest(x_str)

# ax[2].hist(x_fed, color=colors[0], alpha=0.3)
# ax[2].hist(x_str, color=colors[1], alpha=0.3)
# ax[2].set_title(
#     f'Fed: dip stat = {dip_fed:.3f}, p-value = {pval_fed:.4f}\n'
#     f'Starved: dip stat = {dip_str:.3f}, p-value = {pval_str:.4f}'
# )

legend_handles = []
for s, label in enumerate(['fed', 'starved']):
    legend_handles.append(
        mpatches.Patch(color=colors[s], label=label)
    )

fig.legend(handles=legend_handles, loc='center right')

plt.show()