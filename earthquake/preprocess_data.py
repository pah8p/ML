
import csv
import h5py
import numpy

with open('train.csv', newline='') as f:
	n = 0
	reader = csv.reader(f)
	quake = []
	for r, row in enumerate(reader):

		if r == 0: continue
			
		if r % 1000000 == 0: print(n, r)

		try:
			if float(row[1]) > quake[-1][1]:
				new_quake = True
			else:
				new_quake = False
		except IndexError:
			new_quake = False

		if new_quake:
			with h5py.File('quakes.hdf5', 'w') as out:
				out.create_dataset(str(n), data=numpy.array(quake))
			n += 1
			quake = []
			print(n)
	
		quake.append(numpy.array(
			[int(row[0]), float(row[1])],
			#[numpy.int16, numpy.float64],
		))

		
