
from sklearn import linear_model, kernal_ridge, ensemble, metrics, model_selection, pipeline, preprocessing
import pandas
import numpy

class ModelBase(object):

	def __init__(self, model, y, x_train, x_test):
		self.model = model
		self.y = y
		self.x_train = x_train
		self.x_test = x_test

	def cross_validate(self):
		kf = model_selection.KFold(5, shuffle=True, random_state = 42).get_n_splits(self.x_train)
		score = numpy.sqrt(-model_selection.cross_val_score(
			self.model,
			self.x_train,
			self.y,
			scoring = 'neg_mean_squared_error',
			cv = kf
		))
		return score, numpy.mean(score), numpy.stdev(score)

	def fit(self):
		return self._fit()

	def _fit(self):
		#self._x = self.training_data[x].to_numpy()
		#self._y = self.training_data[[y]].to_numpy()
		self.model.fit(self.x_train, self.y)
		#self.x = x
		#self.y = y
		self.y_hat = self.model.predict(self.x_train)
		self.cv = self.cross_validate()

	#def predict(self, keys):
	#	return self._predict(self.testing_data, keys)

	#def errors(self, keys):
	#	data = self._predict(self.training_data, list(set(keys+[self.y])))
	#	data['SqErr'] = (data['PredValue'] - data[self.y])**2
	#	return data

	#def _predict(self, data, keys):
	#	_x = data[self.x].to_numpy()
	#	ys = self.model.predict(_x)
	#	preds = []
	#	for (i, r), y in zip(data.iterrows(), ys):
	#		p = [y[0]]
	#		for key in keys:
	#			p.append(r[key])
	#		preds.append(p)
	#	return pandas.DataFrame(preds, columns=['PredValue']+keys)

class Linear(ModelBase):

	def __init__(self, y, x_train, x_test):
		super().__init__(
			linear_model.LinearRegression(), 
			y, 
			x_train, 
			x_test, 
		)

	def fit(self):
		self._fit()
		self.intercept = self.model.intercept_[0]
		#self.coefficients = dict(zip(x, *self.model.coef_))
		self.r2 = self._r2(self.y, self.y_hat)
		self.mse = self._mse(self.y, self.y_hat)

	def _r2(self, train, pred):
		return metrics.r2_score(train, pred)

	def _mse(self, train, pred):
		return metrics.mean_squared_error(train, pred)
			

class Lasso(ModelBase):

	def __init__(self, y, x_train, x_test, alpha=None):
		lasso = linear_model.Lasso(alpha=alpha)
		model = pipeline.make_pipeline(preprocessing.RobustScaler(), lasso)
		super().__init__(
			model,
			y,
			x_train,
			x_test,
		)

	def fit(self):
		self._fit()
		#self.r2 = self._r2()

	def _r2(self):
		self.model.score()

class ElasticNet(ModelBase):

	def __init__(self, y, x_train, x_test):
		elastic_net = linear_model.ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)
		model = pipeline.make_pipeline(preprocessing.RobustScaler(), elastic_net)
		super().__init__(
			model,
			y,
			x_train,
			x_test,
		)

class KernalRidge(ModelBase):
	
	def __init__(self, y, x_train, x_test):
		model = kernel_ridge.KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
		super().__init__(
			model,
			y,	
			x_train,
			x_test,
		)

class GradientBoosting(ModelBase):

	def __init__(self, y, x_train, x_test):
		model = ensemble.GradientBoostingRegressor(
			n_estimators=3000, 
			learning_rate=0.05,
			max_depth=4, 
			max_features='sqrt',
			min_samples_leaf=15, 
			min_samples_split=10, 
			loss='huber', 
			random_state =5
		)
		super().__init__(
			model,
			y,
			x_train,
			x_test,
		)

