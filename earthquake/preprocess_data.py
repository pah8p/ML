
import csv
import h5py
import numpy


def fourier_transform():

	with h5py.File('quakes.hdf5') as quakes:

		for n in ['0']: #quakes:

			f = numpy.fft.fft(quakes[n][:,1])
			e = numpy.fft.fftfreq(quakes[n].shape[0], quakes[n][0,1]-quakes[n][1,1])

			with h5py.File('fourier.hdf5', 'a') as out:
				try:
					out.create_dataset(n, data=numpy.array(list(zip(f, e))))
				except RuntimeError:
					del out[n]
					out.create_dataset(n, data=numpy.array(list(zip(f, e))))

def csv_to_hdf5():

	with open('train.csv', newline='') as f:
		n = 0
		reader = csv.reader(f)
		quake = []

		for r, row in enumerate(reader):

			if r == 0: 
				continue
				
			if r % 1000000 == 0: 
				print(n, r)

			try:
				if float(row[1]) > quake[-1][1]:
					new_quake = True
				else:
					new_quake = False
			except IndexError:
				new_quake = False

			if new_quake:
				with h5py.File('quakes.hdf5', 'a') as out:
					if str(n) not in out:
						out.create_dataset(str(n), data=numpy.array(quake))
				n += 1
				quake = []
				print(n)
		
			quake.append(numpy.array(
				[int(row[0]), float(row[1])],
			))

fourier_transform()
		
