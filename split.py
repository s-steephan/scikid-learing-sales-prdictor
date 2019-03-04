from pandas import Series
series = Series.from_csv('appsinfinity.csv', header=0)
split_point = len(series) - 12
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset_training.csv')
validation.to_csv('dataset_validation.csv')