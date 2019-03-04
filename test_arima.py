from pandas import Series
from statsmodels.tsa.arima_model import ARIMAResults
import numpy
from matplotlib import pyplot

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

predictions = list()
series = Series.from_csv('dataset_training.csv')
months_in_year = 12
model_fit = ARIMAResults.load('sales_model.pkl')
bias = numpy.load('model_bias.npy')
yhat = float(model_fit.forecast()[0])
predictions.append(yhat)
for i in range(1, 60):
	yhat = bias + inverse_difference(series.values, yhat, months_in_year)
	predictions.append(yhat)

pyplot.plot(predictions, color='red')
pyplot.show()