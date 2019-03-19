
import pandas
import plot
import numpy
import movie_data
from scipy import stats, special
import regression

x_train = pandas.read_csv('train.csv')
x_test = pandas.read_csv('test.csv')
test_id = x_test['id']


x_train['log_revenue'] = numpy.log1p(x_train['revenue'])
x_train['sqrt_revenue'] = x_train['revenue']**(1/5)
x_train['bc_revenue'] = special.boxcox1p(x_train['revenue'], 0.15)

#print(stats.kstest(x_train['bc_revenue'], 'norm'))

plot.view([
	#[plot.fitted_histogram, x_train['popularity']],
	#[plot.fitted_histogram, x_train['sqrt_revenue']],
	#[plot.qq, x_train['bc_revenue']],
])

y = x_train['sqrt_revenue']
y_np = y.to_numpy()

#x = pandas.concat((x_train, x_test), sort=False).reset_index(drop=True)
#movies, x = movie_data.build(x)
#x.to_csv('x.csv', index=False)

x = pandas.read_csv('x.csv')

#print(x.isnull().sum().sort_values(ascending=False).head())

#x['popularity'] = numpy.log1p(x['popularity'])
x['popularity'] = special.boxcox1p(x['popularity'], 0.1)
#x['budget'] = numpy.log1p(x['budget'])

#plot.view([
#	[plot.fitted_histogram, x['budget']],
#	[plot.qq, x['budget']],
#])


x_train_np = x[:3000].to_numpy()
x_test_np = x[3000:].to_numpy()

#x_train.to_csv('x_train.csv', index=False)
#x_test.to_csv('x_test.csv', index=False)

#x_train_np = pandas.read_csv('x_train.csv').to_numpy()
#x_test_np = pandas.read_csv('x_test.csv').to_numpy()


#linear = regression.build('Linear')
#linear_cv = regression.cross_validate(linear, cleaner.x_train_np, y_np)
#print('LINEAR', linear_cv)

lasso = regression.build('Lasso', alpha=0.002)
#lasso_cv = regression.cross_validate(lasso, x_train_np, y_np)
#print('LASSO', lasso_cv)

elastic_net = regression.build('ElasticNet', alpha=0.002)
#elastic_net_cv = regression.cross_validate(elastic_net, x_train_np, y_np)
#print('ELASTIC NET', elastic_net_cv)

kernel_ridge = regression.build('KernelRidge')
#kernel_ridge_cv = regression.cross_validate(kernel_ridge, x_train_np, y_np)
#print('KERNEL RIDGE', kernel_ridge_cv)

gradient_boost = regression.build('GradientBoosting')
#gdcv = regression.cross_validate(gradient_boost, x_train_np, y_np)
#print('GRADIENT BOOST', gdcv)

xg_boost = regression.build(
	'XGBoost', 
	gamma=0.025, 
	max_depth=4, 
	min_child_weight=1.5, 
	subsample=0.5, 
	colsample_bytree=0.5,
	reg_lambda=0.75,
	reg_alpha=0.40,
	n_estimators=2000,
	learning_rate=0.01,
)
xg_cv = regression.cross_validate(xg_boost, x_train_np, y_np)
print('XG BOOST', xg_cv[1])

subs = [gradient_boost, xg_boost] 
model = regression.build('ElasticNet', alpha=0.005)
stacked = regression.build('Stacked', model=model, sub_models=subs)
#stacked_cv = regression.cross_validate(stacked, x_train_np, y_np)
#print('STACKED', stacked_cv)


xg_boost.fit(x_train_np, y_np)
plot.scatter(x_train['revenue'], xg_boost.predict(x_train_np)**5)

res = pandas.DataFrame()
res['id'] = test_id
res['revenue'] = xg_boost.predict(x_test_np)**5
res.to_csv('predictions3.csv', index=False)



