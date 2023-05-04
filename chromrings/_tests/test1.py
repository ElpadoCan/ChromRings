import pandas as pd

series = pd.Series(data=[1,2,3,4,5,6,7,8,9], name='a')

sampling_values = [1,4,7]

print(series[series.isin(sampling_values)])
