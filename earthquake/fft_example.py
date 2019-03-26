
import numpy
from matplotlib import pyplot

def f(t): return numpy.cos(6*numpy.pi*t)*numpy.exp(-numpy.pi*t**2)

ts = [t/1000 for t in range(-2000, 2001)]
fs = [f(t) for t in ts]

ps = numpy.fft.fft(fs)
es = numpy.fft.fftfreq(len(ts), ts[1]-ts[0])

_ps, _es = [], []
for p, e in zip(ps, es):
	if abs(e) <= 6:
		_ps.append(p)
		_es.append(e)


pyplot.subplot(3, 1, 1)
pyplot.plot(ts, fs, color='firebrick')

pyplot.subplot(3, 1, 2)
pyplot.scatter(_es, numpy.abs(_ps), color='royalblue')


pyplot.subplot(3, 1, 3)
pyplot.plot(ts, numpy.fft.ifft(ps), color='seagreen')


pyplot.show()
