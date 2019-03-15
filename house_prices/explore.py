
import datetime
import pandas
import math
from sklearn import linear_model, metrics, preprocessing
from matplotlib import pyplot

def category_values(df, var_name):
	return list(set(df[var_name].to_list()))

def category_to_boolean(df, var_name):
	#categories = list(set(df[var_name].to_list()))
	booleans = pandas.get_dummies(df[var_name])
	df.drop(var_name, axis=1)
	df = df.join(booleans)
	return df #, categories

def clean_numerical(x):
	if math.isnan(x):
		return 0
	return x

def clean_data(data):
	data['TotalBsmtSF'] = data['TotalBsmtSF'].map(lambda x: clean_numerical(x))
	data['TotalSF'] = data['1stFlrSF'] + data['2ndFlrSF'] #+ data['TotalBsmtSF']
	# data['DateSold'] = data.apply(lambda x: datetime.date(x['YrSold'], x['MoSold'], 1), axis = 1)
#	data['OverallQual'] = data['OverallQual'].map(lambda x: quality_map[x])
	data['OverallCond'] = data['OverallCond'].map(lambda x: quality_map[x])

	data = category_to_boolean(data, 'Neighborhood')
	#data, house_styles = category_to_boolean(data, 'HouseStyle')
	#data, zones = category_to_boolean(data, 'MSZoning')
	#data = category_to_boolean(data, 'OverallQual')
	#data = category_to_boolean(data, 'SaleCondition')
	data = category_to_boolean(data, 'OverallCond')

	return data

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
blind_test = pandas.read_csv('test.csv')

# data.drop('column_name', axis=1, inplace=True)

data = clean_data(data)

neighborhoods = category_values(data, 'Neighborhood')
qualities = category_values(data, 'OverallCond')
#conditions = category_values(data, 'SaleCondition')

blind_test = clean_data(blind_test)
#for i, r in blind_test.iterrows():
#	if r['Id'] == 2121:
#		print(r['TotalSF'], type(r['TotalBsmtSF']), clean_numerical(r['TotalBsmtSF']))
#print(blind_test)


prices = data[['SalePrice']].to_numpy()

# print(data)
# print(neighborhoods)
# print(qualities)

dep_names = [
#	'OverallQual', 
#	'OverallCond',
#	'1stFlrSF', 
#	'2ndFlrSF',
#	'YrSold',
#	'TotalSF',
#	'YearRemodAdd',
#	'DateSold',
#	'YearBuilt',
#	'BedroomAbvGr',
]

dep_names += neighborhoods
#dep_names += house_styles
#dep_names += zones
dep_names += qualities
#dep_names += conditions

#for name in dep_names:
#	print(name)
#	print([x for x in blind_test[name].to_list() if not x])

deps = data[dep_names].to_numpy()
blind_deps = blind_test[dep_names].to_numpy()

#print(blind_deps)

split = len(data)//2
split = 0

if split > 0:
	x_train = deps[-split:]
	x_test = deps[:-split]
	y_train = prices[-split:]
	y_test = prices[:-split]
else:
	x_train = deps
	x_test = deps
	y_train = prices
	y_test = prices


model = linear_model.LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
coefs = dict(zip(dep_names, *model.coef_))
intercept = model.intercept_[0]

print(metrics.r2_score(y_test, y_pred))
print(metrics.mean_squared_error(y_test, y_pred))
#print(dict(zip(dep_names, *model.coef_)))


my_pred = predict(coefs, intercept, data)
print(sum([x[2] for x in my_pred])/len(my_pred))
pandas.DataFrame(my_pred, columns=['Id', 'SalePrice', 'Err']).to_csv('sampe.csv', index=False)


#blind_pred = model.predict(blind_deps)
blind_pred = predict(coefs, intercept, blind_test)
blind_preds = pandas.DataFrame(blind_pred, columns=['Id', 'SalePrice'])
blind_preds.to_csv('predictions.csv', index=False)

if len(dep_names) == 1:
	pyplot.scatter(x_train, y_train, color='black')
	pyplot.plot(x_train, y_pred, color='blue', linewidth=3)
	pyplot.show()


