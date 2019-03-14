
import pandas
import plot
import numpy
import regression
from clean import Cleaner


def map_1_10(x):
	_map = {
		10: 'VeryExcellent',
		9: 'Excellent',
		8: 'VeryGood',
		7: 'Good',
		6: 'AboveAverage',
		5: 'Average',
		4: 'BelowAverage',
		3: 'Fair',
		2: 'Poor',
		1: 'VeryPoor',
	}
	return _map[x]

training_data = pandas.read_csv('train.csv', na_filter=False)
testing_data = pandas.read_csv('test.csv', na_filter=False)

numeric_vars = [
#	{'name': '1stFlrSF'},
#	{'name': 'YrSold'},
#	{'name': 'YearBuilt'},
#	{'name': 'BedroomAbvGr'},
	{'name': 'TotalSF', 'sum': ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF']},
	{'name': 'OverallRating', 'sum': ['OverallQual', 'OverallCond']},
]

category_vars = [
#	{'name': 'OverallQual', 'func': map_1_10},
#	{'name': 'OverallCond', 'func': map_1_10},
	{'name': 'Neighborhood'},
#	{'name': 'HouseStyle'},
#	{'name': 'MSZoning'},
#	{'name': 'SaleCondition'},
#	{'name': 'SaleType'},
	{'name': 'Exterior1st'},
	{'name': 'Condition1'},
	{'name': 'LotShape'},
	{'name': 'LandContour'},
]

training_data['Log1SalePrice'] = numpy.log(1+training_data['SalePrice'])

plots = [
	[plot.fitted_histogram, training_data['SalePrice']],
	[plot.fitted_histogram, training_data['Log1SalePrice']],
	[plot.qq, training_data['SalePrice']],
	[plot.qq, training_data['Log1SalePrice']],
]
#plot.view(plots)
	

cleaner = Cleaner(training_data, testing_data)

#print(cleaner.category_values(cleaner.training_data, 'Exterior1st'))
#print(cleaner.category_values(cleaner.testing_data, 'Exterior1st'))

#print(len(training_data))
#print(len(testing_data))

cleaner.clean(numeric_vars, category_vars)

#print(len(cleaner.training_data))
#print(len(cleaner.testing_data))

model = regression.Linear(cleaner.training_data, cleaner.testing_data)



#print(cleaner.variables)

model.fit(cleaner.variables, 'SalePrice')

print(model.r2)
print(model.mse)

#model.errors(
#	['Id', 'SalePrice', 'SaleType', 'BldgType', 'Exterior1st', 'Neighborhood', 'OverallQual',
#	'Condition1','OverallCond']
#).to_csv('errors.csv', index=False)

#model.predict(['Id']).to_csv('predict.csv', index=False)



