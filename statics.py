from pandas import Series
series = Series.from_csv('dataset_training.csv')
print(series.describe())
