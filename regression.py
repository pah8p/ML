
from sklearn import linear_model, metrics
import pandas

class ModelBase(object):

	def __init__(self, model, training_data, testing_data):
		self.model = model()
		self.training_data = training_data
		self.testing_data = testing_data

	def _fit(self, x, y):
		self._x = self.training_data[x].to_numpy()
		self._y = self.training_data[[y]].to_numpy()
		self.model.fit(self._x, self._y)
		self.x = x
		self.y = y
		self.y_hat = self.model.predict(self._x)

class Linear(ModelBase):

	def __init__(self, training_data, testing_data):
		super().__init__(
			linear_model.LinearRegression, 
			training_data, 
			testing_data
		)

	def fit(self, x, y):
		#_x = self.training_data[x].to_numpy()
		#_y = self.training_data[[y]].to_numpy()
		#self.model.fit(_x, _y)
		#pred = self.model.predict(_x)
		self._fit(x, y)
		self.intercept = self.model.intercept_[0]
		self.coefficients = dict(zip(x, *self.model.coef_))
		self.r2 = self._r2(self._y, self.y_hat)
		self.mse = self._mse(self._y, self.y_hat)
		#self.x = x
		#self.y = y

	def _r2(self, train, pred):
		return metrics.r2_score(train, pred)

	def _mse(self, train, pred):
		return metrics.mean_squared_error(train, pred)

	def predict(self, keys):
		return self._predict(self.testing_data, keys)

	def errors(self, keys):
		data = self._predict(self.training_data, keys)
		data['SqErr'] = (data['PredValue'] - data[self.y])**2
		return data

	def _predict(self, data, keys):
		_x = data[self.x].to_numpy()
		ys = self.model.predict(_x)
		preds = []
		for (i, r), y in zip(self.training_data.iterrows(), ys):
			p = [y[0]]
			for key in keys:
				p.append(r[key])
			preds.append(p)
		return pandas.DataFrame(preds, columns=['PredValue']+keys)
			

class Lasso(object):

	def __init__(self, training_data, testing_data):
		super().__init__(
			linear_model.Lasso,
			training_data,
			testing_data,
		)
