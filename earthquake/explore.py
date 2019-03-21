
import numpy
import pandas
import seaborn
from matplotlib import pyplot


seaborn.set_style('darkgrid')

N = 50*10**6

chunks = pandas.read_csv(
	'train.csv',
	#nrows = N, 
	chunksize = N, 
	dtype = {
		'acoustic_data': numpy.int16,
		'time_to_failure': numpy.float32,
	}
)

for n, chunk in enumerate(chunks):
	print(n)
	numpy.savez_compressed('numpy_data/chunk%s.npz' % n, data = chunk.to_numpy())

#print(x.memory_usage(index=True).sum())
#print(x.head())




'''
n = range(N)

fig, ax1 = pyplot.subplots()

ax1.set_xlabel('n')
ax1.set_ylabel('acoustic_data')
ax1.plot(n, x['acoustic_data'], color='tab:blue')
ax1.tick_params(axis='y')

ax2 = ax1.twinx() 

ax2.set_ylabel('time_to_failure')  
ax2.plot(n, x['time_to_failure'], color='tab:red')
ax2.tick_params(axis='y')

fig.tight_layout()  
pyplot.show()
'''
