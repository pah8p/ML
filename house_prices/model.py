
import pandas
import plot
import numpy
import regression
from clean import Cleaner
import warnings

def ignore_warn(*args, **kwargs):
    pass

warnings.warn = ignore_warn

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
	{'name': 'GarageCars', 'na_val': 0},
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

x_train = x_train.drop(x_train[(x_train['GrLivArea']>4000) & (x_train['SalePrice']<300000)].index)

y = pandas.DataFrame(x_train['SalePrice'], columns=['SalePrice'])

prices = x_train['SalePrice']
x_train.drop('SalePrice', axis=1, inplace=True)

y['LogSalePrice'] = numpy.log(y['SalePrice'])

plots = [
	[plot.fitted_histogram, y['SalePrice']],
	[plot.fitted_histogram, y['LogSalePrice']],
#	[plot.qq, y['SalePrice']],
#	[plot.qq, y['Log1SalePrice']],
]
#plot.view(plots)

#y.to_csv('y.csv', index=False)
y_np = y.drop('SalePrice', axis=1).to_numpy()

train_id = x_train['Id']
test_id = x_test['Id']

x_train.drop('Id', axis=1, inplace=True)
x_test.drop('Id', axis=1, inplace=True)

cleaner = Cleaner(x_train, x_test)
cleaner.clean(variables)

linear = regression.build('Linear')
#linear_cv = regression.cross_validate(linear, cleaner.x_train_np, y_np)
#print('LINEAR', linear_cv)

lasso = regression.build('Lasso', alpha=0.002)
#lasso_cv = regression.cross_validate(lasso, cleaner.x_train_np, y_np)
#print('LASSO', lasso_cv)

elastic_net = regression.build('ElasticNet', alpha=0.002)
#elastic_net_cv = regression.cross_validate(elastic_net, cleaner.x_train_np, y_np)
#print('ELASTIC NET', elastic_net_cv)

kernel_ridge = regression.build('KernelRidge')
#kernel_ridge_cv = regression.cross_validate(kernel_ridge, cleaner.x_train_np, y_np)
#print('KERNEL RIDGE', kernel_ridge_cv)

gradient_boost = regression.build('GradientBoosting')
#gdcv = regression.cross_validate(gradient_boost, cleaner.x_train_np, y_np)
#print('GRADIENT BOOST', gdcv)

xg_boost = regression.build(
	'XGBoost', 
	gamma=0.0468, 
	max_depth=3, 
	min_child_weight=1.7817, 
	subsample=0.5213, 
	colsample_bytree=0.4603,
	reg_lambda=0.8571,
	reg_alpha=0.4640,
	n_estimators=2200,
	learning_rate=0.05,
)
#xg_cv = regression.cross_validate(xg_boost, cleaner.x_train_np, y_np)
#print('XG BOOST', xg_cv)

subs = [lasso, elastic_net, kernel_ridge, gradient_boost, xg_boost] 
model = regression.build('Lasso', alpha=0.005)
stacked = regression.build('Stacked', model=model, sub_models=subs)
#stacked_cv = regression.cross_validate(stacked, cleaner.x_train_np, y_np)
#print('STACKED', stacked_cv)

stacked.fit(cleaner.x_train_np, y_np)
plot.scatter(prices, numpy.exp(stacked.predict(cleaner.x_train_np)))

#res = pandas.DataFrame()
#res['Id'] = test_id
#res['SalePrice'] = numpy.exp(stacked.predict(cleaner.x_test_np))
#res.to_csv('predictions3.csv', index=False)




