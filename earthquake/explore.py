
import numpy
import pandas
import seaborn
from matplotlib import pyplot
from scipy import signal
import h5py
import utils

seaborn.set_style('darkgrid')

N = 27176955
#N = 10*10**6

quakes = h5py.File('quakes.hdf5', 'r')

print(quakes['6'][0:N,0].shape)
quake = quakes['6']


#t = chunks.iloc[0]['time_to_failure']-chunks['time_to_failure']
with utils.Watch('Subtraction'):
	t = quake[0, 1]-quake[0:N,1]

#y = chunks['acoustic_data']


def fourier_transform(t, y):
	return (
		numpy.fft.fftfreq(len(t), t[1]-t[0]),
		numpy.fft.fft(y),
	)

def hi_filter(y_hat, eta, threshold):
	return [y if abs(e) < threshold else 0 for y, e in zip(y_hat, eta)]

def plot(charts):
	for n, chart in enumerate(charts):
		pyplot.subplot(len(charts), 1, n+1)
		pyplot.plot(chart[0], chart[1], color=chart[2])
	pyplot.show()

with utils.Watch('Fourier'):
	eta, y_hat = fourier_transform(t, quake[0:N,0])


#hi = hi_filter(y_hat, eta, 0.01)

plot([
	[quake[0:N,1], quake[0:N,0], 'navy'],
	[eta, numpy.abs(y_hat), 'firebrick'],
	[eta, y_hat.real, 'seagreen'],
	[eta, y_hat.imag, 'lawngreen'],
#	[eta, numpy.abs(hi), 'deeppink'],
])


'''
a, b, c = signal.spectrogram(y, 1/(t[1]-t[0]))
pyplot.pcolormesh(b, a, c)
pyplot.show()
'''


