import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from chromrings import tables_path, core

X = np.linspace(0,100,1000)

print(np.std(X))

model = core.PeaksModel(n_peaks=1)

x0_1 = 1
sx_1 = 1
max_1 = 1
A_1 = max_1 * sx_1 * np.sqrt(2*np.pi)

x0_2 = 8
sx_2 = 1
max_2 = 1
A_2 = max_2 * sx_2 * np.sqrt(2*np.pi)

# coeffs = [x0_1, sx_1, A_1, x0_2, sx_2, A_2]
coeffs = [74.19764901, 14.53905668, 22.58263259]

Y = model.model(X, coeffs, 0)

plt.plot(X, Y)
plt.show()

exit()

df = pd.DataFrame.from_dict({'Col1': [5, 20, 25, 100], 'Col2': [10, 20, 30, 40]})
df = df / df.max()

print(df.max())

print(df)

exit()

s = pd.Series(data=[1,2,3,4])

print(s)

s2 = pd.Series(data=[1,3,5])

print(s2)

df = pd.concat([s, s2], axis=1)

print(df)

exit()

df = pd.DataFrame({
    'a': np.random.randint(1,30,100)
})

print(df)

df.index = pd.to_datetime(df.index)

print(df)

resampled = df.resample('10ns', label='right').mean()

print(resampled)

print(resampled.index.astype(np.int64))

exit()

x1, y1 = 1, 2

xx = [1, 2, 1, 3]
yy = [2, 2, 3, 3]

arr = np.column_stack((xx, yy))

print(arr)

diff = arr - (x1, y1)

print(diff)

dist = np.linalg.norm(diff, axis=1)

print(dist)

df = pd.DataFrame(index=dist)

# print(df)

df['value_0'] = np.random.randint(1,20,len(dist))
df['value_1'] = np.random.randint(1,20,len(dist))

print(df)

print('--------------')

df_mean = df.mean(axis=1)

print(df_mean)