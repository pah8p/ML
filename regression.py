
from sklearn import linear_model, kernel_ridge, ensemble
from sklearn import metrics, model_selection, pipeline, preprocessing, base
import pandas
import numpy

class ModelBase(object):

	def __init__(self, model):
		self.model = model

	def cross_validate(self, x, y):
		kf = model_selection.KFold(5, shuffle=True, random_state = 42).get_n_splits(x)
		score = numpy.sqrt(-model_selection.cross_val_score(
			self.model,
			x,
			y,
			scoring = 'neg_mean_squared_error',
			cv = kf
		))
		return score, numpy.mean(score)

	def fit(self, x, y):
		return self._fit(x, y)

	def _fit(self, x, y):
		self.model.fit(x, y)
		self.y_hat = self.model.predict(x)
		self.cv = self.cross_validate(x, y)

	def predict(self, x):
		return self.model.predict(x)

	def _r2(self, train, pred):
		return metrics.r2_score(train, pred)

	def mse(self, train, pred):
		return metrics.mean_squared_error(train, pred)

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

class Average(ModelBase):
	pass

class Stacked(object):

	def __init__(self, model, sub_models):
		self.sub_models = sub_models
		self.model = model

	def fit(self, x, y):

		fitted_models = []

		kf = model_selection.KFold(n_splits = 5, shuffle=True, random_state = 42)
		
		preds = numpy.zeros((len(x), len(self.sub_models)))

		for n, sub_model in enumerate(self.sub_models):
			fitted_submodels = []
			for train, test in kf.split(x, y):
				
				sub_model.fit(x[train], y[train])
				pred = sub_model.predict(x[test])
	
				try:
					if pred.shape[1] == 1:
						pred = pred[0]	
				except:
					pass

				preds[test, n] = pred

				fitted_submodels.append(sub_model)

			fitted_models.append(fitted_submodels)

		self.model.fit(preds, y)
		self.fitted_models = fitted_models

	def predict(self, x):

		preds = []
		for fitted_submodels in self.fitted_models:
			_preds = []
			for submodel in fitted_submodels:
				_preds.append(submodel.predict(x))
			preds.append(numpy.column_stack(_preds).mean(axis=1))
	
		_x = numpy.column_stack(preds)
		return self.model.predict(_x)



class Linear(ModelBase):

	def __init__(self): #, y, x_train, x_test):
		super().__init__(
			linear_model.LinearRegression(), 
			#y, 
			#x_train, 
			#x_test, 
		)

	def fit(self):
		self._fit()
		self.intercept = self.model.intercept_[0]
		#self.coefficients = dict(zip(x, *self.model.coef_))
		self.r2 = self._r2(self.y, self.y_hat)
		self.mse = self._mse(self.y, self.y_hat)


			

class Lasso(ModelBase):

	def __init__(self, alpha=1.0):
		lasso = linear_model.Lasso(alpha=alpha, max_iter=1000)
		model = pipeline.make_pipeline(preprocessing.RobustScaler(), lasso)
		super().__init__(
			model,
			#y,
			#x_train,
			#x_test,
		)

	def fit(self, x, y):
		self._fit(x, y)
		#self.r2 = self._r2()

	def _r2(self):
		self.model.score()

class ElasticNet(ModelBase):

	def __init__(self, alpha=1.0):
		elastic_net = linear_model.ElasticNet(alpha=alpha, l1_ratio=.9, random_state=3, max_iter=1000)
		model = pipeline.make_pipeline(preprocessing.RobustScaler(), elastic_net)
		super().__init__(
			model,
		)

class KernelRidge(ModelBase):
	
	def __init__(self):
		model = kernel_ridge.KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
		super().__init__(
			model,
		)

class GradientBoosting(ModelBase):

	def __init__(self):
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
			model
		)

