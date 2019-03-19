
import sklearn
from sklearn import linear_model, kernel_ridge, ensemble
from sklearn import metrics, model_selection, pipeline, preprocessing
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


class Stacked(sklearn.base.BaseEstimator):

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
			'sub_models': [],
		},
	},
	'XGBoost': {
		'model': xgboost.XGBRegressor,
		'kwargs': {
			'booster': 'gbtree',       	# Can be gbtree, gblinear or dart; gbtree and dart use tree based models while gblinear uses linear functions.
			'verbosity': 0,			# Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug)
			'learning_rate': 0.1,		# Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative			
			'gamma': 0,			# Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be 			
			'max_depth': 6, 	 	# Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit
			'min_child_weight': 1,		# Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be.			
			'colsample_bytree': 1, 		# The subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
			'n_estimators': 100,		# Number of trees to fit
			'reg_alpha': 0, 		# L1 regularization term on weights. Increasing this value will make model more conservative.
			'reg_lambda': 1,		# L2 regularization term on weights. Increasing this value will make model more conservative.
			'subsample': 1, 		# Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. Subsampling will occur once in every boosting iteration.
#			'random_state': 7, 
			'nthread': -1,			# Number of parallel threads used to run XGBoost
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

def xgboost_features(model, x_df):
	#features = []
	#for feature, importance in zip(x.columns.values.tolist(), xg_boost.feature_importances_):
	#	features.append({
	#		'feature': feature,
	#		'importance': importance,
	#	})	

	#features.sort(key = lambda x: x['importance'], reverse = True)

	#import seaborn
	#from matplotlib import pyplot
	#seaborn.barplot(data = pandas.DataFrame(features).head(10), x='feature', y='importance')
	#pyplot.show()
	pass
