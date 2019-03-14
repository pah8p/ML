
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

x_train = pandas.read_csv('train.csv', na_filter=False)
x_test = pandas.read_csv('test.csv', na_filter=False)

numeric_vars = [
#	{'name': '1stFlrSF'},
#	{'name': 'YrSold'},
#	{'name': 'YearBuilt'},
#	{'name': 'BedroomAbvGr'},
	{'name': 'TotalSF', 'sum': ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF']},
#	{'name': 'OverallRating', 'sum': ['OverallQual', 'OverallCond']},
]

category_vars = [
	{'name': 'OverallQual', 'func': map_1_10},
#	{'name': 'OverallCond', 'func': map_1_10},
	{'name': 'Neighborhood'},
#	{'name': 'HouseStyle'},
#	{'name': 'MSZoning'},
#	{'name': 'SaleCondition'},
#	{'name': 'SaleType'},
#	{'name': 'Exterior1st'},
#	{'name': 'Condition1'},
#	{'name': 'LotShape'},
#	{'name': 'LandContour'},
]


y = pandas.DataFrame(x_train['SalePrice'], columns=['SalePrice'])
x_train.drop('SalePrice', axis=1, inplace=True)

y['Log1SalePrice'] = numpy.log1p(y['SalePrice'])

plots = [
	[plot.fitted_histogram, y['SalePrice']],
	[plot.fitted_histogram, y['Log1SalePrice']],
	[plot.qq, y['SalePrice']],
	[plot.qq, y['Log1SalePrice']],
]
#plot.view(plots)
y.to_csv('y.csv', index=False)
y_np = y.drop('SalePrice', axis=1).to_numpy()

cleaner = Cleaner(x_train, x_test)
cleaner.clean(numeric_vars, category_vars)

cleaner.x_train.to_csv('x_train.csv', index=False)

linear_model = regression.Linear(y, cleaner.x_train_np, cleaner.x_test_np)
linear_model.fit()
print(linear_model.cv)

lasso_model = regression.Lasso(y, cleaner.x_train_np, cleaner.x_test_np, 0.005)
lasso_model.fit()
print(lasso_model.cv)

#print(model.y_hat)
#print(model.r2)
#print(model.r2)
#print(model.mse)

#model.errors(
#	['Id', 'SalePrice', 'SaleType', 'BldgType', 'Exterior1st', 'Neighborhood', 'OverallQual',
#	'Condition1','OverallCond']
#).to_csv('errors.csv', index=False)

#model.errors(['Id']).to_csv('predict.csv', index=False)



