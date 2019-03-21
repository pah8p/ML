
from matplotlib import pyplot
from scipy import stats
import seaborn
import numpy
import itertools
import pandas

seaborn.set_style('darkgrid')

def view(plots):
	for i, (plot, data) in enumerate(plots):
		pyplot.figure(i)		
		plot(data)
	pyplot.show()

def scatter(x, y):

	df = pandas.DataFrame()
	df['x'] = x
	df['y'] = y

	slope, intercept, r_value, p_value, std_err = stats.linregress(df['x'], df['y'])
	seaborn.regplot(x='x', y='y', data=df, line_kws = {'label': 'Slope=%s Intercept=%s R2=%s' % (slope, intercept, r_value)})
	#seaborn.scatterplot(x=x, y=y)
	pyplot.show()

def fitted_histogram(data):
	seaborn.distplot(data, fit=stats.norm)


def qq(data):
	fig = pyplot.figure()
	res = stats.probplot(data, plot=pyplot)
	
	
def correlation_matrix(data):
	matrix = data.corr()
	seaborn.heatmap(matrix, vmax=0.9, square=True, cmap = pyplot.cm.RdYlGn)

def bar(data):
	seaborn.barplot(data)

def confusion_matrix(matrix):
	seaborn.heatmap(matrix, cmap = pyplot.cm.Blues, annot=True, square=True, x='Predicted', y='Actual')
	pyplot.show()

def show_image(img):
	pyplot.figure()
	pyplot.imshow(img)
	pyplot.colorbar()
	pyplot.grid(False)
	pyplot.show()
