
import numpy
import pandas
import seaborn
from matplotlib import pyplot
from scipy import signal
import h5py
import utils

seaborn.set_style('darkgrid')


with utils.Watch('load'):
	quakes = h5py.File('quakes.hdf5', 'r')
	fourier = h5py.File('fourier.hdf5', 'r')

#with utils.Watch('keys'):
#	print(quakes.keys())

#with utils.Watch('shapes'):
#	for k in quakes:
#		print (k, quakes[k].shape)

#t = chunks.iloc[0]['time_to_failure']-chunks['time_to_failure']

#with utils.Watch('Subtraction'):
#	t = quake[0, 1]-quake[0:N,1]

#y = chunks['acoustic_data']


def fourier_transform(t, y):
	return (
		numpy.fft.fftfreq(len(t), t[1]-t[0]),
		numpy.fft.fft(y),
	)

def hi_filter(y_hat, eta, threshold):
	return [y if abs(e) < threshold else 0 for y, e in zip(y_hat, eta)]

def ot_filter(y_hat, eta, hi, lo):
	return numpy.fft.ifft([y if (e < lo and e > hi) else 0 for y, e in zip(y_hat, eta)])

def in_filter(y_hat, eta, hi, lo):
	return numpy.fft.ifft([y if (e > lo and e < hi) else 0 for y, e in zip(y_hat, eta)])

def plot(charts, save=None):
	for n, chart in enumerate(charts):
		pyplot.subplot(len(charts), 1, n+1)
		pyplot.plot(chart[0], chart[1], color=chart[2])
	if save:
		pyplot.savefig(save)
	else:
		pyplot.show()

#with utils.Watch('Fourier'):
#	eta, y_hat = fourier_transform(t, quake[0:N,0])


#hi = hi_filter(y_hat, eta, 0.01)

#out_filter(fourier['0'][:,0], fourier['0'][:,1], 0, 0)

print(fourier['0'][1:,1])
print(numpy.abs(fourier['0'][1:100,1]), numpy.abs(fourier['0'][1:100,0]))

F = 10e2

plot([
	[quakes['0'][:,1], quakes['0'][:,0], 'navy'],
#	[numpy.abs(fourier['0'][1:100,1]), numpy.abs(fourier['0'][1:100,0]), 'firebrick'],
	[quakes['0'][:,1], numpy.abs(in_filter(fourier['0'][:,1], fourier['0'][:,0], F, -F)), 'seagreen'],
	[quakes['0'][:,1], numpy.abs(ot_filter(fourier['0'][:,1], fourier['0'][:,0], F, -F)), 'deeppink'],
], 'plots/fourier_0.png')


'''
a, b, c = signal.spectrogram(y, 1/(t[1]-t[0]))
pyplot.pcolormesh(b, a, c)
pyplot.show()
'''


