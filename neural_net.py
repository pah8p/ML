
import numpy

class SimpleNeuralNet(object):

	def __init__(self, x, y, num_nodes, learning_rate):
		self.num_nodes = num_nodes
		self.learning_rate = learning_rate
		self.x = x
		self.y = y
		self.w = [
			numpy.random.standard_normal((x.shape[1], num_nodes))*(2.0/(x.shape[1]-num_nodes))**0.5,
			numpy.array([numpy.random.random() for n in range(num_nodes)])*(2.0/(num_nodes-1))**0.5,
		]

	def _relu(self, x):
		#return numpy.array([_x if _x > 0 else 0.01 for _x in x])
		return numpy.array([max(z, 0.1) for z in x])

	def _predict(self, x):
		return self._relu(x @ self.w[0]) @ self.w[1] 
		
	def predict(self):
		return numpy.array([self._predict(z) for z in x])

	def _squared_error(self, x, y):
		return 0.5*(self._predict(x)-y)**2

	def squared_error(self):
		return numpy.array([self._squared_error(_x, _y) for _x, _y in zip(self.x, self.y)])

	def mean_squared_error(self):
		return numpy.mean(self.squared_error())

	def _update(self, x, y):
		
		dE_dYhat = self._predict(x)-y
		dYhat_dH = self.w[1]
		dYhat_dw1 = self._relu(x @ self.w[0])

		dH_dw0 = numpy.zeros((x.shape[0], self.num_nodes))
		theta = x @ self.w[0]		
		for i in range(len(x)):
			for n in range(self.num_nodes):
				if theta[n] > 0:
					dH_dw0[i, n] = x[i]
				else:
					dH_dw0[i, n] = 0.1*x[i]

		E = self.mean_squared_error()
		
		factors = [len(x)*self.num_nodes/self.learning_rate, self.num_nodes/self.learning_rate]
	
		w = [
			self.w[0] - E/(dE_dYhat * dYhat_dH * dH_dw0)/factors[0],
			self.w[1] - E/(dE_dYhat * dYhat_dw1)/factors[1],
		]

		self.w = w

	def fit(self, epochs):
		print(0, self.mean_squared_error())
		for epoch in range(epochs):
			for x, y in zip(self.x, self.y):	
				self._update(x, y)
			print(epoch+1, self.mean_squared_error())


x = numpy.array([
	[1,2,3,4,5],
	[6,7,8,9,10],
	[11,12,13,14,15],
	[16,17,18,19,20],
])

y = numpy.array([6, 21, 36, 51])

#print(x.shape)

snn = SimpleNeuralNet(x, y, 4, 0.5)

print(snn.w)

#print(snn._predict(x[1]))
#print(snn.predict())
#print(snn.squared_error())
#print(snn.mean_squared_error())

#print(snn._predict(x[0]))
#for i in range(5):
#	snn._update(x[0], y[0])

snn.fit(5)

print(snn.w)

