
import pandas
import plot
import numpy
import movie_data
from scipy import stats, special
import regression

x_train = pandas.read_csv('train.csv')
x_test = pandas.read_csv('test.csv')
test_id = x_test['id']

features = [
	{'name': 'belongs_to_collection', 'drop': True},	
	{'name': 'budget', 'na_val': 0}, #TODO this is wrong
	{'name': 'genres', 'drop': True},
	{'name': 'homepage', 'drop': True},
	{'name': 'imdb_id', 'drop': True},
	{'name': 'original_langage', 'drop': True},
	{'name': 'original_title', 'na_val': 'None'},
	{'name': 'overview', 'drop': True},
	{'name': 'popularity'},
	{'name': 'poster_path', 'drop': True},
	{'name': 'production_companies', 'drop': True},
	{'name': 'production_countries', 'drop': True},
	{'name': 'release_date'},
	{'name': 'runtime'},
	{'name': 'spoken_languages', 'drop': True},
	{'name': 'status', 'drop': True},
	{'name': 'tagline', 'drop': True},
	{'name': 'title', 'drop': True},
	{'name': 'Keywords', 'drop': True},
	{'name': 'cast', 'drop': True},
	{'name': 'crew', 'drop': True},

	{'name': 'collection', 'na_val': 'None'},
	{'name': 'adventure', 'na_val': 'None'},
	{'name': 'animation', 'na_val': 'None'},
	{'name': 'crime', 'na_val': 'None'},
	{'name': 'horror', 'na_val': 'None'},
	{'name': 'comedy', 'na_val': 'None'},
	{'name': 'romance', 'na_val': 'None'},
	{'name': 'drame', 'na_val': 'None'},
	{'name': 'foreign', 'na_val': 'None'},
	{'name': 'war', 'na_val': 'None'},
	{'name': 'science_fiction', 'na_val': 'None'},
	{'name': 'family', 'na_val': 'None'},
	{'name': 'thriller', 'na_val': 'None'},
	{'name': 'action', 'na_val': 'None'},
	{'name': 'western', 'na_val': 'None'},
	{'name': 'music', 'na_val': 'None'},
	{'name': 'history', 'na_val': 'None'},
	{'name': 'tv_movie', 'na_val': 'None'},
	{'name': 'documentary', 'na_val': 'None'},
	{'name': 'fantasy', 'na_val': 'None'},
	{'name': 'mystery', 'na_val': 'None'},
	{'name': 'language', 'na_val': 'None'},
	{'name': 'director', 'na_val': 'None'},
]


x_train['log_revenue'] = numpy.log1p(x_train['revenue'])
x_train['sqrt_revenue'] = x_train['revenue']**(1/5)
x_train['bc_revenue'] = special.boxcox1p(x_train['revenue'], 0.15)

#print(stats.kstest(x_train['bc_revenue'], 'norm'))

#plot.view([
	#[plot.fitted_histogram, x_train['revenue']],
	#[plot.fitted_histogram, x_train['sqrt_revenue']],
	#[plot.qq, x_train['bc_revenue']],
#])

y = x_train['sqrt_revenue']
y_np = y.to_numpy()

x = pandas.concat((x_train, x_test), sort=False).reset_index(drop=True)
movies, x = movie_data.build(x)

print(x.isnull().sum().sort_values(ascending=False).head())

x_train_np = x[:3000].to_numpy()
x_test_np = x[3000:].to_numpy()
x.to_csv('x.csv', index=False)
#x_train.to_csv('x_train.csv', index=False)
#x_test.to_csv('x_test.csv', index=False)

#x_train_np = pandas.read_csv('x_train.csv').to_numpy()
#x_test_np = pandas.read_csv('x_test.csv').to_numpy()


#linear = regression.build('Linear')
#linear_cv = regression.cross_validate(linear, cleaner.x_train_np, y_np)
#print('LINEAR', linear_cv)

lasso = regression.build('Lasso', alpha=0.002)
lasso_cv = regression.cross_validate(lasso, x_train_np, y_np)
print('LASSO', lasso_cv)

elastic_net = regression.build('ElasticNet', alpha=0.002)
elastic_net_cv = regression.cross_validate(elastic_net, x_train_np, y_np)
print('ELASTIC NET', elastic_net_cv)

kernel_ridge = regression.build('KernelRidge')
kernel_ridge_cv = regression.cross_validate(kernel_ridge, x_train_np, y_np)
print('KERNEL RIDGE', kernel_ridge_cv)

#gradient_boost = regression.build('GradientBoosting')
#gdcv = regression.cross_validate(gradient_boost, cleaner.x_train_np, y_np)
#print('GRADIENT BOOST', gdcv)

xg_boost = regression.build(
	'XGBoost', 
	gamma=0.02, 
	max_depth=3, 
	min_child_weight=1.7817, 
	subsample=0.5, 
	colsample_bytree=0.5,
	reg_lambda=0.8571,
	reg_alpha=0.4640,
	n_estimators=2200,
	learning_rate=0.05,
)
#xg_cv = regression.cross_validate(xg_boost, x_train_np, y_np)
#print('XG BOOST', xg_cv)

subs = [lasso, elastic_net, kernel_ridge, xg_boost] 
model = regression.build('Lasso', alpha=0.005)
stacked = regression.build('Stacked', model=model, sub_models=subs)
stacked_cv = regression.cross_validate(stacked, x_train_np, y_np)
print('STACKED', stacked_cv)

#xg_boost.fit(x_train_np, y_np)

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

#print(xg_boost.feature_importances_)
plot.scatter(x_train['revenue'], stacked.predict(x_train_np)**5)

#res = pandas.DataFrame()
#res['id'] = test_id
#res['revenue'] = xg_boost.predict(x_test_np)**5
#res.to_csv('predictions.csv', index=False)



