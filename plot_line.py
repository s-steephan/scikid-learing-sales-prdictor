from pandas import Series
from matplotlib import pyplot
series = Series.from_csv('dataset_training.csv')
series.plot()
pyplot.show()