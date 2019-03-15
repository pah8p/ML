
import pandas
import plot
import numpy
import clean
import ast
import expand

x_train = pandas.read_csv('train.csv')
x_test = pandas.read_csv('test.csv')

x = ast.literal_eval(x_train.iloc[0, 1])
print(type(x))

#print(x_train.iloc[0, 1])

expand.expand_columns(x_test)

genres = []

for r in x_train.iterrows():
	try:
		gs = ast.literal_eval(r[1]['spoken_languages'])
		for g in gs:
			genres.append(g['name'])
	except:
		pass

for r in x_test.iterrows():
	try:
		gs = ast.literal_eval(r[1]['spoken_languages'])
		for g in gs:
			genres.append(g['name'])
	except:
		pass

for z in set(genres): print(z)
