from Tkinter import *
import tkMessageBox

from pandas import Series
from matplotlib import pyplot

root = Tk()

root.geometry("500x500")
root.resizable(False, False)

def plot_line():
	series = Series.from_csv('dataset.csv')
	series.plot()
	pyplot.show()
	# tkMessageBox.showinfo( "Hello Python", "Hello World")

label = Label( root, text="dsafjasdk fsd f sdfj" )
label.grid(column=0, row=0)

B = Button(root, text ="Plot Line Graph", command = plot_line)
B.grid(column=1, row=0)

B.pack()
label.pack()
root.mainloop()