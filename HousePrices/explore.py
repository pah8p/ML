
import datetime
import pandas
from sklearn import linear_model, metrics, preprocessing
from matplotlib import pyplot

def category_to_boolean(df, var_name):
	categories = list(set(df[var_name].to_list()))
	booleans = pandas.get_dummies(df[var_name])
	df.drop(var_name, axis=1)
	df = df.join(booleans)
	return df, categories

quality_map = {
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

data = pandas.read_csv('train.csv')

# data.drop('column_name', axis=1, inplace=True)

prices = data[['SalePrice']].to_numpy()

data['TotalSF'] = data['1stFlrSF'] + data['2ndFlrSF'] + data['TotalBsmtSF']
# data['DateSold'] = data.apply(lambda x: datetime.date(x['YrSold'], x['MoSold'], 1), axis = 1)
data['OverallQual'] = data['OverallQual'].map(lambda x: quality_map[x])


data, neighborhoods = category_to_boolean(data, 'Neighborhood')
data, house_styles = category_to_boolean(data, 'HouseStyle')
data, zones = category_to_boolean(data, 'MSZoning')
data, qualities = category_to_boolean(data, 'OverallQual')

# print(data)
# print(neighborhoods)
# print(qualities)

dep_names = [
#	'OverallQual', 
#	'OverallCond',
#	'1stFlrSF', 
#	'2ndFlrSF',
#	'YrSold',
	'TotalSF',
#	'DateSold',
	'YearBuilt',
	'BedroomAbvGr',
]

dep_names += neighborhoods
#dep_names += house_styles
#dep_names += zones
dep_names += qualities

deps = data[dep_names].to_numpy()

model = linear_model.LinearRegression()

model.fit(deps, prices)

model_prices = model.predict(deps)

print(metrics.r2_score(prices, model_prices))
print(metrics.mean_squared_error(prices, model_prices))
print(dict(zip(dep_names, *model.coef_)))


if len(dep_names) == 1:
	pyplot.scatter(deps, prices, color='black')
	pyplot.plot(deps, model_prices, color='blue', linewidth=3)
	pyplot.show()


