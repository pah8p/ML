
import pandas
import math

class Cleaner(object):

	def __init__(self, data):
		self.data = data

	def category_values(self, category):
		return list(set(self.data[category].to_list()))

	def category_to_bool(self, category):
		dummies = pandas.get_dummies(self.data[category])
		self.data.drop(category, axis=1)
		self.data = self.data.join(dummies)

	def clean_column(self, name, f):
		self.data[name] = self.data[name].map(lambda x: f(x))

	def clean_numeric(self, x):
		if math.isnan(x):
			return 0
		return x

	def clean(self, numerics, categories):
	
		self.variables = []

		for numeric in numerics:
			try:
				_f = numeric['func']
			except KeyError:
				_f = self.clean_numeric
			n = numeric['name']			
			self.clean_column(n, _f)
			self.variables.append(n)

		for category in categories:
			c = category['name']
			
			if 'func' in category:
				self.clean_column(c, category['func'])

			for cv in self.category_values(c):
				self.variables.append(cv)

			self.category_to_bool(c)
				


