
from matplotlib import pyplot
from scipy import stats
import seaborn

seaborn.set_style('darkgrid')

def view(plots):
	for i, (plot, data) in enumerate(plots):
		pyplot.figure(i)		
		plot(data)
	pyplot.show()

def scatter(x, y):
	seaborn.scatterplot(x=x, y=y)
	pyplot.show()

def fitted_histogram(data):
	seaborn.distplot(data, fit=stats.norm)


def qq(data):
	fig = pyplot.figure()
	res = stats.probplot(data, plot=pyplot)
	
	
def correlation_matrix(data):
	matrix = data.corr()
	pyplot.subplots(figsize=(12,9))
	seaborn.heatmap(matrix, vmax=0.9, square=True)

def bar(data):
	seaborn.barplot(data)
	
