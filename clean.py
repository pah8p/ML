
import pandas
import math
import numpy
from sklearn import preprocessing
from scipy import stats, special

class Cleaner(object):

	def __init__(self, x_train, x_test):
		self.x_train = x_train
		self.x_test = x_test

	def nan_count(self, data):

		nans = data.isnull().sum()
		
		if sum(nans) == 0:
			return 'No NANs!'
		else:
			return pandas.DataFrame({'NAN_%': (data.isnull().sum()/len(data)) * 100})		

	def box_cox_transform(self, data):
		numericals = data.dtypes[data.dtypes != 'object'].index
		skewed = pandas.DataFrame({'Skew': data[numericals].apply(lambda x: stats.skew(x.dropna()))})
		big_skew = skewed[abs(skewed) > 0.75]
		big_skew_features = big_skew.index
		lam = 0.15
		for feature in big_skew_features:
			data[feature] = special.boxcox1p(data[feature], lam)
		return data
		
	def sum_columns(self, data, name, cols):
		data[name] = data.apply(
			lambda r: sum([r[col] for col in cols]),
			axis=1
		)
		return data

	def clean_column(self, data, col):
		name = col['name']
		
		if 'na_val' in col:
			data[name] = data[name].fillna(col['na_val'])

		elif 'na_f' in col:

			if col['na_f'] == 'mode':
				data[name] = data[name].fillna(data[name].mode()[0])

			elif col['na_f'] == 'median':
				data[name] = data[name].fillna(data[name].median()[0])

			elif col['na_f'] == 'average':
				data[name] = data[name].fillna(data[name].average()[0])			

			elif col['na_f'] == 'attr_median':
				data[name] = data.groupby(col['na_f_args']['groupby'])[name].transform(lambda x: x.fillna(x.median()))

		return data


	def clean(self, variables):
	
		split = len(self.x_train)
		
		data = pandas.concat((self.x_train, self.x_test), sort=False).reset_index(drop=True)

		print(data.shape)

		print(self.nan_count(data))

		for var in variables:

			name = var['name']

			if 'drop' in var:
				if var['drop']:
					data = data.drop([name], axis=1)
					break

			if 'sum' in var:
				data = self.sum_columns(data, name, var['sum'])
	
			if 'str' in var:
				data[name] = data[name].astype(str)
				#data = self.clean_column(data, c, category['rename'])

			data = self.clean_column(data, var)

			if 'label_encode' in var:
				encoder = preprocessing.LabelEncoder()
				encoder.fit(data[name].to_list())
				data[name] = encoder.transform(data[name].to_list())
			
		data = self.box_cox_transform(data)		
		data = pandas.get_dummies(data)

		print(data.shape)
		print(self.nan_count(data))

		self.x_train = data[:split]
		self.x_test = data[split:]
		
		self.x_train_np = self.x_train.to_numpy()
		self.x_test_np = self.x_test.to_numpy()
		
