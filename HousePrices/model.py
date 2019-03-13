
import pandas
from kaggle.clean import Cleaner
from kaggle.linear import LinearRegression

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

training_data = pandas.read_csv('train.csv')
testing_data = pandas.read_csv('test.csv')

numeric_vars = [
	{'name': '1stFlrSF'},
	{'name': 'YrSold'},
	{'name': 'YearBuilt'},
	{'name': 'BedroomAbvGr'},
	{'name': 'TotalSF', 'agg': ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF']},
]

category_vars = [
	{'name': 'OverallQual', 'func': map_1_10},
	{'name': 'OverallCond', 'func': map_1_10},
	{'name': 'Neighborhood'},
	{'name': 'HouseStyle'},
	{'name': 'MSZoning'},
	{'name': 'SaleCondition'},
]

training_cleaner = Cleaner(training_data)
training_cleaner.clean(numeric_vars, category_vars)

testing_cleaner = Cleaner(testing_data)
testing_cleaner.clean(numeric_vars, category_vars)

model = LinearRegression(training_cleaner.data)

print(model.r2)
print(model.mse)



