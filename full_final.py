from Tkinter import *
import tkMessageBox

from pandas import Series, DataFrame, TimeGrouper
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
import numpy

window = Tk()
window.title("Sales Predictor")	
window.geometry('500x500')
window.resizable(False, False)

def split_dataset():
	series = Series.from_csv('appsinfinity.csv', header=0)
	split_point = len(series) - 12
	dataset, validation = series[0:split_point], series[split_point:]
	print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
	dataset.to_csv('dataset_training.csv')
	validation.to_csv('dataset_validation.csv')
	x = str(len(dataset))
	y = str(len(validation))

	tkMessageBox.showinfo("Info", 'Dataset %s, Validation %s' % (x, y))

def plot_line():
	series = Series.from_csv('dataset_training.csv')
	series.plot()
	pyplot.show()

def plot_test_line_graph():
	series = Series.from_csv('dataset_validation.csv')
	pyplot.figure(num='Graph from validation Dataset')
	series.plot()
	pyplot.show()

def plot_seasonal_graph():
	series = Series.from_csv('dataset_training.csv')
	groups = series['2010':'2016'].groupby(TimeGrouper('A'))
	years = DataFrame()
	pyplot.figure()
	i = 1
	n_groups = len(groups)
	for name, group in groups:
		pyplot.subplot((n_groups*100) + 10 + i)
		i += 1
		pyplot.plot(group)
	pyplot.show()

def plot_density_graph():
	series = Series.from_csv('dataset_training.csv')
	pyplot.figure(1)
	pyplot.subplot(211)
	series.hist()
	pyplot.subplot(212)
	series.plot(kind='kde')
	pyplot.show()
	lbl = Label(window, text="Sales Predictor")
	lbl.place(x=210, y=10, anchor=NW)

def describe_dataset():
	series = Series.from_csv('dataset_training.csv')
	print(series.describe())
	str_dataset_details = str(series.describe())
	tkMessageBox.showinfo("Input Dataset Details", str_dataset_details)

def __getnewargs__(self):
	return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))

ARIMA.__getnewargs__ = __getnewargs__

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

def train():
	series = Series.from_csv('dataset_training.csv')
	X = series.values
	X = X.astype('float32')
	months_in_year = 12
	diff = difference(X, months_in_year)
	model = ARIMA(diff, order=(0,0,1))
	model_fit = model.fit(trend='nc', disp=0)
	bias = 165.904728
	# save model
	model_fit.save('sales_model.pkl')
	numpy.save('model_bias.npy', [bias])
	tkMessageBox.showinfo("Training Done", "Training completed. <<< sales_model.pkl >>> created successfully.")

ery_monts = Entry(window)
ery_monts.place(x=200, y=350, anchor=NW)

def predict():
	try:
		months = int(ery_monts.get())
		if months <= 0:
			tkMessageBox.showinfo("Error", "Months should be greater than zero")
		else:
			dataset = Series.from_csv('dataset_training.csv')
			X = dataset.values.astype('float32')
			history = [x for x in X]
			months_in_year = 12

			model_fit = ARIMAResults.load('sales_model.pkl')
			bias = numpy.load('model_bias.npy')
			predictions = list()
			yhat = float(model_fit.forecast()[0])
			yhat = bias + inverse_difference(history, yhat, months_in_year)
			predictions.append(yhat)
			history.append(X[0])

			for i in range(0, months):
				iteration = months 
				diff = difference(history, iteration)

				# predict here
				model = ARIMA(diff, order=(0,0,1))
				model_fit = model.fit(trend='nc', disp=0)
				yhat = model_fit.forecast()[0]
				yhat = bias + inverse_difference(history, yhat, iteration)

				#add predicted data in predictions list to plot graph
				predictions.append(yhat)


				#add predicted data to history
				#which helps to improve accuracy while predict future
				history.append(yhat)
				print("Prediction> "+ str(yhat))

			pyplot.figure(num='Predicted Results')
			pyplot.plot(predictions, color='red')
			pyplot.show()
	except Exception as e:
		tkMessageBox.showinfo("Error", "invalid month")

label = Label(window, text="Sales Predictor" )
label.place(x=200, y=10, anchor=NW)

btn_split_dataset = Button(window, text="Split Dataset", command=split_dataset)
btn_split_dataset.place(x=100, y=60, anchor=NW)
 
btn_plot_line_graph = Button(window, text="Plot Line Graph", command=plot_line)
btn_plot_line_graph.place(x=300, y=60, anchor=NW)

btn_seasonal_plot = Button(window, text="Seasonal Line Plots", command=plot_seasonal_graph)
btn_seasonal_plot.place(x=100, y=120, anchor=NW)

btn_density_plot = Button(window, text="Density Plot", command=plot_density_graph)
btn_density_plot.place(x=300, y=120, anchor=NW)

btn_describe_dataset = Button(window, text="Describe Dataset", command=describe_dataset)
btn_describe_dataset.place(x=100, y=180, anchor=NW)

btn_train = Button(window, text="Train", command=train)
btn_train.place(x=100, y=300, anchor=NW)

btn_validate = Button(window, text="Validate", command=plot_test_line_graph)
btn_validate.place(x=300, y=300, anchor=NW)

btn_describe_dataset = Button(window, text="Predict", command=predict)
btn_describe_dataset.place(x=240, y=380, anchor=NW)


window.mainloop()