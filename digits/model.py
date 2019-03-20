
import regression
import pandas
import numpy

#N = 20000

x_train = pandas.read_csv('train.csv')#, nrows=N)
y = x_train['label']
x_train = x_train.drop(['label'], axis=1)

def gt0(x):
	if x > 0:
		return 1
	else:
		return 0

#x_train.applymap(gt0)

def normalize(row):

	colored = []
	for i in range(27):
		for j in range(27):
			n = 28*i + j
			if row['pixel%s' % n]:
				colored.append((i, j))

	min_i = min(colored, key=lambda x: x[0])[0]
	max_i = max(colored, key=lambda x: x[0])[0]
	min_j = min(colored, key=lambda x: x[1])[1]
	max_j = max(colored, key=lambda x: x[1])[1]

	#return [(min_i, min_j), (min_i, max_j), (max_i, min_j), (max_i, max_j)]
	return [max_i-min_i, max_j-min_j]

#for n, row in x_train.head(100).iterrows():
#	print(n, normalize(row))
			
				



#split = int(N/2)

#x_test_np = x_train[split:].to_numpy()
#y_test_np = y[split:].to_numpy()



#logistic = regression.build('Logistic', C=0.1)
#logistic_cv = regression.cross_validate(logistic, x_train_np, y_np, scoring='completeness_score')
#print('LOGISTIC', logistic_cv[1])
#logistic.fit(x_train_np, y_np)
#print(logistic.score(x_test_np, y_test_np))

svc = regression.build('SVC', C=0.1, kernel='poly', degree=2, cache_size=500)
svc.fit(x_train.to_numpy(), y.to_numpy())

x_train = pandas.read_csv('test.csv')
assert(len(x_train)==28000)

pred = pandas.DataFrame()
pred['ImageId'] = list(range(1, 28000+1))
pred['Label'] = svc.predict(x_train.to_numpy())
pred.to_csv('predictions.csv', index=False)

#print(svc.score(x_test, y_test_np))


#def score(model, x, y):
#	return sum([1*(y==y_pred) for y, y_pred in zip(y, model.predict(x))])/split

#print(score(logistic, x_test_np, y_test_np))









