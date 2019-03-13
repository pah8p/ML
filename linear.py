
from sklearn import linear_model, metrics
import pandas

class LinearRegression(object):

	def __init__(self, training_data):
		self.model = linear_model.LinearRegression()
		self.training_data = training_data

	def fit(self, x, y):
		_x = self.training_data[x].to_numpy()
		_y = self.training_data[[y]].to_numpy()
		self.model.fit(_x, _y)
		y = self.model.predict(_x)
		self.intercept = self.model.intercept_[0]
		self.coefficients = dict(zip(x, *self.model.coef_))
		self.r2 = self._r2(_y, y)
		self.mse = self._mse(_y, y)

	def _r2(self, train, pred):
		return metrics.r2_score(train, pred)

	def _mse(self, train, pred):
		return metrics.mean_squared_error(train, pred)

	def predict(self, x, keys):
		ys = self.model.predict(x)
		preds = []
		for (i, r), y in zip(self.training_data.iterrows(), ys):
			p = [y]
			for key in keys:
				p.append(r[key])
			preds.append(p)
		return pandas.DataFrame(preds, columns=['PredValue']+keys)
			


# def predict(coefs, intercept, data):
# 	predictions = []
#	for index, row in data.iterrows():
#		p = intercept
#		for k, v in coefs.items():
#			p += row[k]*v
#
#		if 'SalePrice' in row:
#			predictions.append((row['Id'], p, (p-row['SalePrice'])**2))
#		else:
#			predictions.append((row['Id'], p))
#	
#	return predictions

