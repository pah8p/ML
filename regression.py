
from sklearn import linear_model, kernel_ridge, ensemble
from sklearn import metrics, model_selection, pipeline, preprocessing, base
import pandas
import numpy
import xgboost


def cross_validate(model, x, y, n_folds=5):
	kf = model_selection.KFold(n_folds, shuffle=True, random_state = 42).get_n_splits(x)
	score = numpy.sqrt(-model_selection.cross_val_score(
		model,
		x,
		y,
		scoring = 'neg_mean_squared_error',
		cv = kf
	))
	return score, numpy.mean(score)


class Stacked(object):

	def __init__(self, model, sub_models):
		self.sub_models = sub_models
		self.model = model

	def fit(self, x, y):

		fitted_models = []

		kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
		
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

MODELS = {
	'Linear': {
		'model': linear_model.LinearRegression,
		'kwargs': {},
	},
	'Lasso': {
		'model': linear_model.Lasso,
		'kwargs': {'alpha': 0.005},
		'preprocessing': preprocessing.RobustScaler,
	},
	'ElasticNet': {
		'model': linear_model.ElasticNet,
		'kwargs': {
			'alpha': 0.005, 
			'l1_ratio': 0.9, 
			'random_state': 3, 
			'max_iter': 1000,
		},
		'preprocessing': preprocessing.RobustScaler,
	},
	'KernelRidge': {
		'model': kernel_ridge.KernelRidge,
		'kwargs': {
			'alpha': 0.6, 
			'kernel': 'polynomial', 
			'degree': 2, 
			'coef0': 2.5,
		},
	},
	'GradientBoosting': {
		'model': ensemble.GradientBoostingRegressor,
		'kwargs': {
			'n_estimators': 3000, 
			'learning_rate': 0.05,
			'max_depth': 4, 
			'max_features': 'sqrt',
			'min_samples_leaf': 15, 
			'min_samples_split': 10, 
			'loss': 'huber', 
			'random_state': 5,
		},
	},
	'Stacked': {
		'model':  Stacked,
		'kwargs': {
			'model': None,
			'submodels': [],
		},
	},
	'XGBoost': {
		'model': xgboost.XGBoostRegressor,
		'kwargs': {
			'colsample_bytree': 0.4603, 
			'gamma': 0.0468, 
			'learning_rate': 0.05, 
			'max_depth': 3, 
			'min_child_weight': 1.7817, 
			'n_estimators': 2200,
			'reg_alpha': 0.4640, 
			'reg_lambda': 0.8571,
			'subsample': 0.5213, 
			'silent': 1,
			'random_state': 7, 
			'nthread': -1,
		},
	},
}

def build(model_type, **kwargs):

	config = MODELS[model_type]

	_kwargs = config['kwargs']

	for k, v in kwargs.items():
		_kwargs[k] = v

	model = config['model'](**_kwargs)

	if 'preprocessing' in config:
		preprocessor = config['preprocessing']()
		model = pipeline.make_pipeline(preprocessor, model)

	return model





