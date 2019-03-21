
import regression
import pandas
import numpy
import utils
import plot
import tensorflow
from tensorflow import keras
from sklearn import model_selection, metrics

N = 500

with utils.Watch('Loading data'):
	x = pandas.read_csv('train.csv', nrows=N)

y = x['label'].to_numpy()
x = x.drop(['label'], axis=1)

x /= 255.0
x = x.to_numpy().reshape(N, 28, 28, 1)


x_train, x_test, y_train, y_test = model_selection.train_test_split(
	x, y, test_size = 0.1, random_state=2
)

def support_vector_machine(x, y, predict=False):

	x = x.reshape(len(x), 28*28)

	svc = regression.build('SVC', C=0.1, kernel='poly', degree=2, cache_size=500)
	svc.fit(x, y)
	print(svc.score(x, y))

	if predict:
		x_test = pandas.read_csv('test.csv')

		pred = pandas.DataFrame()
		pred['ImageId'] = list(range(1, 28000+1))
		pred['Label'] = svc.predict(x_test.to_numpy())
		pred.to_csv('predictions.csv', index=False)

#support_vector_machine(x_train, y_train)

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
	tf.fit(x_train, y_train, epochs=2)

with utils.Watch('Evaluating model'):
	print(tf.evaluate(x_test, y_test))

with utils.Watch('Generating confusion matrix'):
	y_pred = [numpy.argmax(p) for p in tf.predict(x_test)]
	matrix = metrics.confusion_matrix(y_test, y_pred)
	plot.confusion_matrix(matrix)


def predict(model):

	test = pandas.read_csv('test.csv')
	test /= 255.0
	test = testx.values.reshape(len(test), 28, 28, 1)
#	test = test.applymap(lambda x: x/255.0)
#	test = numpy.array([expand_image(r) for r in test.to_numpy()])


	with utils.Watch('Predicting test data'):
		pred = pandas.DataFrame()
		pred['ImageId'] = list(range(1, 28000+1))
		pred['Label'] = [numpy.argmax(p) for p in tf.predict(test)]
		pred.to_csv('predictions4.csv', index=False)

#predict(tf)




