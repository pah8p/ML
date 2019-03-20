
import regression
import pandas
import numpy
from matplotlib import pyplot
import tensorflow
from tensorflow import keras
import utils


N = 42000

with utils.Watch('Loading data'):
	x = pandas.read_csv('train.csv', nrows=N)

y = x['label']
x = x.drop(['label'], axis=1)

x = x.applymap(lambda x: x/255.0)

def show_image(img):
	pyplot.figure()
	pyplot.imshow(img)
	pyplot.colorbar()
	pyplot.grid(False)
	pyplot.show()

def expand_image(r):
	image = numpy.zeros((28, 28, 1))
	for i in range(28):
		for j in range(28):
			image[i, j, 0] = r[28*i + j] #/255.0
	return image

with utils.Watch('Transforming data'):
	x_np = numpy.array([expand_image(r) for r in x.to_numpy()])

#split = int(N/2)

#x_train = x_np[:split]
y_np = y.to_numpy()
#x_test = x_np[split:]	
#y_test = y[split:].to_numpy()

#print(x_train.shape)


#logistic = regression.build('Logistic', C=0.1)
#logistic_cv = regression.cross_validate(logistic, x_train_np, y_np, scoring='completeness_score')
#print('LOGISTIC', logistic_cv[1])
#logistic.fit(x_train_np, y_np)
#print(logistic.score(x_test_np, y_test_np))

#svc = regression.build('SVC', C=0.1, kernel='poly', degree=2, cache_size=500)
#svc.fit(x_train.to_numpy(), y_train.to_numpy())
#print(svc.score(x_test.to_numpy(), y_test.to_numpy()))

#x_train = pandas.read_csv('test.csv')
#assert(len(x_train)==28000)

#pred = pandas.DataFrame()
#pred['ImageId'] = list(range(1, 28000+1))
#pred['Label'] = svc.predict(x_train.to_numpy())
#pred.to_csv('predictions.csv', index=False)

#tf = keras.Sequential([
#	keras.layers.Flatten(input_shape=(28, 28)),
#	keras.layers.Dense(128, activation=tensorflow.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001)),
#	keras.layers.Dropout(0.5),
#	keras.layers.Dense(10, activation=tensorflow.nn.softmax),
#])


tf = keras.Sequential([
	keras.layers.Conv2D(
		filters = 32, 
		kernel_size = 3, 
		activation = 'relu',		
		padding = 'same', 
		input_shape = (28, 28, 1), 
	),
	keras.layers.Conv2D(
		filters = 32, 
		kernel_size = 3, 
		activation = 'relu',
	),
	keras.layers.MaxPooling2D(pool_size = 2),
	keras.layers.Dropout(0.25),
	keras.layers.Conv2D(
		filters = 64, 
		kernel_size = 3, 
		padding='same', 
		activation = 'relu'
	),
	keras.layers.Conv2D(
		filters = 64, 
		kernel_size = 3,
		activation = 'relu'
	),
	keras.layers.MaxPooling2D(pool_size = 2),
	keras.layers.Dropout(0.25),
	keras.layers.Flatten(),
	keras.layers.Dense(128, activation = 'relu'),
	keras.layers.Dropout(0.5),
	keras.layers.Dense(10, activation = 'softmax'),
])

tf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

with utils.Watch('Fitting model'):
	tf.fit(x_np, y_np, epochs=10)

#with utils.Watch('Evaluating model'):
#	print(tf.evaluate(x_test, y_test))

x_np = numpy.array([expand_image(r) for r in x.to_numpy()])

test = pandas.read_csv('test.csv')
test = test.applymap(lambda x: x/255.0)
test = numpy.array([expand_image(r) for r in test.to_numpy()])

#assert(len(x)==28000)

with utils.Watch('Predicting test data'):
	pred = pandas.DataFrame()
	pred['ImageId'] = list(range(1, 28000+1))
	pred['Label'] = [numpy.argmax(p) for p in tf.predict(test)]
	pred.to_csv('predictions4.csv', index=False)


#def score(model, x, y):
#	return sum([1*(y==y_pred) for y, y_pred in zip(y, model.predict(x))])/split









