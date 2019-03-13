
import pandas
import math

class Cleaner(object):

	def __init__(self, training_data, testing_data):
		self.training_data = training_data
		self.testing_data = testing_data

	def category_values(self, data, category):
		return list(set(data[category].to_list()))

	def category_to_bool(self, data, category):
		#TODO does this work if training and test data
		#have different category values?
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

	def clean_column(self, data, name, f):
		try:
			data[name] = data[name].map(lambda x: f(x))
		except KeyError:
			#TODO log error or something
			pass
		return data

	def clean_numeric(self, x):
		if math.isnan(x):
			return 0
		return x

	def _clean(self, data, numerics, categories):
	
		self.variables = []

		for numeric in numerics:			
			n = numeric['name']
			self.variables.append(n)

			if 'func' in numeric:
				_f = numeric['func']
			else:
				_f = self.clean_numeric
			data = self.clean_column(data, n, _f)
			
			if 'sum' in numeric:
				data = self.sum_columns(data, n, numeric['sum'])


		for category in categories:
			c = category['name']
			
			if 'func' in category:
				data = self.clean_column(data, c, category['func'])

			for cv in self.category_values(data, c):
				self.variables.append(cv)

			data = self.category_to_bool(data, c)
				
		return data

	def clean(self, numerics, categories):
		self.training_data = self._clean(self.training_data, numerics, categories)
		self.testing_data = self._clean(self.testing_data, numerics, categories)

