

import pandas
from sklearn import linear_model, metrics
from matplotlib import pyplot

data = pandas.read_csv('train.csv')

# data.drop('column_name', axis=1, inplace=True)

prices = data[['SalePrice']].to_numpy()

data['TotalSF'] = data['1stFlrSF'] + data['2ndFlrSF'] + data['TotalBsmtSF']

dep_names = [
#	'OverallQual', 
#	'OverallCond',
	'1stFlrSF', 
#	'2ndFlrSF',
#	'YrSold',
#	'TotalSF',
]

deps = data[dep_names].to_numpy()

model = linear_model.LinearRegression()

model.fit(deps, prices)

model_prices = model.predict(deps)

print(metrics.r2_score(prices, model_prices))
print(metrics.mean_squared_error(prices, model_prices))
print(dict(zip(dep_names, *model.coef_)))


pyplot.scatter(deps, prices, color='black')
pyplot.plot(deps, model_prices, color='blue', linewidth=3)

pyplot.show()


