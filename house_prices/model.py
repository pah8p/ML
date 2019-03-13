
import pandas
from clean import Cleaner
from linear import LinearRegression

def map_1_10(x):
	# TODO: VeryExcellent isn't working?
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

training_data = pandas.read_csv('train.csv')
testing_data = pandas.read_csv('test.csv')

numeric_vars = [
#	{'name': '1stFlrSF'},
#	{'name': 'YrSold'},
#	{'name': 'YearBuilt'},
#	{'name': 'BedroomAbvGr'},
	{'name': 'TotalSF', 'sum': ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF']},
]

category_vars = [
	{'name': 'OverallQual', 'func': map_1_10},
#	{'name': 'OverallCond', 'func': map_1_10},
	{'name': 'Neighborhood'},
#	{'name': 'HouseStyle'},
#	{'name': 'MSZoning'},
#	{'name': 'SaleCondition'},
]

cleaner = Cleaner(training_data, testing_data)

#t = training_data #cleaner.clean_column(training_data, 'OverallCond', map_1_10)
#print(list(set(t['OverallQual'].to_list())))


cleaner.clean(numeric_vars, category_vars)
model = LinearRegression(cleaner.training_data, cleaner.testing_data)

print(cleaner.variables)

model.fit(cleaner.variables, 'SalePrice')

print(model.r2)
print(model.mse)

#model.predict(['Id']).to_csv('predict.csv', index=False)



