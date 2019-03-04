import pandas as pd
from matplotlib import pyplot

series = pd.read_csv('dataset_training.csv')
pyplot.figure(1)
pyplot.subplot(211)
series.hist()
pyplot.subplot(212)
series.plot(kind='kde')
pyplot.show()