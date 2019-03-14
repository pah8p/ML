
import pandas
import math
import numpy

class Cleaner(object):

	def __init__(self, x_train, x_test):
		self.x_train = x_train
		self.x_test = x_test

	def category_values(self, data, category):
		return list(set(data[category].to_list()))

	def category_to_bool(self, data, category):
		dummies = pandas.get_dummies(data[category])
		data.drop(category, axis=1)		
		data = data.join(dummies)
		return data

	def sum_columns(self, data, name, cols):
		data[name] = data.apply(
			lambda r: sum([self.clean_numeric(r[col]) for col in cols]),
			axis=1
		)
		return data

	def clean_column(data, col):
		name = col['name']
		
		if 'val' in col:
			data[name] = data[name].fillna(col['val'])

		elif 'f' in col:

			if f == 'mode':
				data[name] = data[name].fillna(data[name].mode()[0])

			elif f == 'median':
				data[name] = data[name].fillna(data[name].median()[0])

			elif f == 'average':
				data[name] = data[name].fillna(data[name].average()[0])			

			elif f == 'attr_median':
				data[name] = data.groupby(col['f_args']['groupby'])[name].transform(lambda x: x.fillna(x.median()))

		return data


	def clean(self, numerics, categories):
	
		split = len(self.x_train)
		
		data = pandas.concat((self.x_train, self.x_test), sort=False).reset_index(drop=True)

		self.variables = []

		for numeric in numerics:			

			n = numeric['name']
			self.variables.append(n)

			if 'sum' in numeric:
				data = self.sum_columns(data, n, numeric['sum'])
	
			data = self.clean_column(data, numeric)
			
		for category in categories:

			c = category['name']
			
			data = self.clean_column(data, category)

			if 'rename' in category:
				data = self.clean_column(data, c, category['rename'])

			for cv in self.category_values(data, c):
				self.variables.append(cv)

			data = self.category_to_bool(data, c)
		
		data = data[self.variables]
		
		self.x_train = data[:split]
		self.x_test = data[split:]
		
		self.x_train_np = self.x_train.to_numpy()
		self.x_test_np = self.x_test.to_numpy()
		
