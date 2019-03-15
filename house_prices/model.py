
import pandas
import plot
import numpy
import regression
from clean import Cleaner

def map_mssubclass(x):
	_map = {
		20: '1-STORY 1946 & NEWER ALL STYLES',
		30: '1-STORY 1945 & OLDER',
		40: '1-STORY W/FINISHED ATTIC ALL AGES',
		45: '1-1/2 STORY - UNFINISHED ALL AGES',
		50: '1-1/2 STORY FINISHED ALL AGES',
		60: '2-STORY 1946 & NEWER',
		70: '2-STORY 1945 & OLDER',
		75: '2-1/2 STORY ALL AGES',
		80: 'SPLIT OR MULTI-LEVEL',
		85: 'SPLIT FOYER',
		90: 'DUPLEX - ALL STYLES AND AGES',
	       120: '1-STORY PUD (Planned Unit Development) - 1946 & NEWER',
	       150: '1-1/2 STORY PUD - ALL AGES',
	       160: '2-STORY PUD - 1946 & NEWER',
	       180: 'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER',
	       190: '2 FAMILY CONVERSION - ALL STYLES AND AGES',
	}
	return str(x) #_map[x]

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

x_train = pandas.read_csv('train.csv')
x_test = pandas.read_csv('test.csv')

variables = [
	{'name': '1stFlrSF'},
	{'name': 'YrSold', 'label_encode': True, 'str': True},
	{'name': 'MoSold', 'label_encode': True, 'str': True},
	{'name': 'YearBuilt', 'label_encode': True, 'str': True},
	{'name': 'BedroomAbvGr'},
	{'name': 'LotFrontage', 'na_f': 'attr_median', 'na_f_args': {'groupby': 'Neighborhood'}},
	{'name': 'GarageYrBlt', 'na_val': 0},
	{'name': 'GarageArea', 'na_val': 0},
	{'name': 'GarageCars', 'na_va': 0},
	{'name': 'BsmtFinSF1', 'na_val': 0},
	{'name': 'BsmtFinSF2', 'na_val': 0},
	{'name': 'BsmtUnfSF', 'na_val': 0},
	{'name': 'TotalBsmtSF', 'na_val': 0},
	{'name': 'BsmtFullBath', 'na_val': 0},
	{'name': 'BsmtHalfBath', 'na_val': 0},
	{'name': 'MasVnrArea', 'na_val': 0},
	{'name': 'TotalSF', 'sum': ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF']},
	{'name': 'OverallQual', 'str': True, 'label_encode': True},
	{'name': 'OverallCond', 'str': True, 'label_encode': True},
	{'name': 'Neighborhood'},
	{'name': 'HouseStyle'},
	{'name': 'SaleCondition'},
	{'name': 'SaleType'},
	{'name': 'Condition1'},
	{'name': 'LandContour'},
	{'name': 'PoolQC', 'na_val': 'None', 'label_encode': True},
	{'name': 'MiscFeature', 'na_val': 'None'},
	{'name': 'Alley', 'na_val': 'None', 'label_encode': True},
	{'name': 'Fence', 'na_val': 'None', 'label_encode': True},
	{'name': 'FireplaceQu', 'na_val': 'None', 'label_encode': True},
	{'name': 'GarageType', 'na_val': 'None'},
	{'name': 'GarageFinish', 'na_val': 'None', 'label_encode': True},
	{'name': 'GarageQual', 'na_val': 'None', 'label_encode': True},
	{'name': 'GarageCond', 'na_val': 'None', 'label_encode': True},
	{'name': 'BsmtQual', 'na_val': 'None', 'label_encode': True},
	{'name': 'BsmtCond', 'na_val': 'None', 'label_encode': True},
	{'name': 'BsmtExposure', 'na_val': 'None', 'label_encode': True},
	{'name': 'BsmtFinType1', 'na_val': 'None', 'label_encode': True},
	{'name': 'BsmtFinType2', 'na_val': 'None', 'label_encode': True},
	{'name': 'MasVnrType', 'na_val': 'None'},
	{'name': 'MSZoning', 'na_f': 'mode'},
	{'name': 'Utilities', 'drop': True},
	{'name': 'Functional', 'na_val': 'Typ'},
	{'name': 'Electrical', 'na_f': 'mode'},
	{'name': 'KitchenQual', 'na_f': 'mode'},
	{'name': 'Exterior1st', 'na_f': 'mode'},
	{'name': 'Exterior2nd', 'na_f': 'mode'},
	{'name': 'SaleType', 'na_f': 'mode'},
	{'name': 'MSSubClass', 'na_val': 'None', 'str': True, 'label_encode': True},
	{'name': 'ExterQual', 'label_encode': True},
	{'name': 'ExterCond', 'label_encode': True},
	{'name': 'HeatingQC', 'label_encode': True},
	{'name': 'KithenQual', 'label_encode': True},
	{'name': 'Functional', 'label_encode': True},
	{'name': 'LandSlope', 'label_encode': True},
	{'name': 'PavedDrive', 'label_encode': True},
	{'name': 'Street', 'label_encode': True},
	{'name': 'CentralAir', 'label_encode': True},
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

#y.to_csv('y.csv', index=False)
y_np = y.drop('SalePrice', axis=1).to_numpy()

cleaner = Cleaner(x_train, x_test)
cleaner.clean(variables)

#cleaner.x_train.to_csv('x_train.csv', index=False)

linear_model = regression.Linear(y, cleaner.x_train_np, cleaner.x_test_np)
linear_model.fit()
print(linear_model.cv)

lasso = regression.Lasso(y, cleaner.x_train_np, cleaner.x_test_np, 0.005)
lasso.fit()
print('LASSO', lasso.cv)

elastic_net = regression.ElasticNe(y, cleaner.x_train_np, cleaner.x_test_np)
elastic_net.fit()
print('ELASTIC NET', elastic_net.cv)

kernel_ridge = regression.KernelRidge(y, cleaner.x_train_np, cleaner.x_test_np)
kernel_ridge.fit()
print('KERNEL RIDGE', kernel_ridge.cv)

gradient_boost = regression.GradientBoosting(y, cleaner.x_train_np, cleaner.x_test_np)
gradient_boost.fit()
print('GRADIENT BOOTS', gradient_boost.cv)


