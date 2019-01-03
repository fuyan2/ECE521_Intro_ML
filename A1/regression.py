import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# calculate euclidean distance square
def D_eucl_sqr(X, Z):
	if X.shape[1] != Z.shape[1]:
		print("D_eucl_sqr: X and Z must have the same number of column")
		return tf.constant([])
	return tf.reduce_sum(tf.square(X[:, None] - Z[None, :]), 2)

# find k nearest neighbors, return responsility matrix
def K_NN(X, Z, k):
	if X.shape[0] < k:
		print("k exceeds the maximum allowed number by matrix!")
		return None
	distance = D_eucl_sqr(Z, X)
	reciprocal = tf.reciprocal(distance)
	values, indices = tf.nn.top_k(reciprocal, k)
	kthValue = tf.reduce_min(values)
	# calculate condition mask
	cond = tf.greater_equal(tf.transpose(reciprocal), kthValue)
	# assign responsibility with 1 or 0
	R = tf.where(cond, tf.ones(cond.shape, tf.float64), tf.zeros(cond.shape, tf.float64))
	# normalize responsibilty to sum up to 1
	R = tf.divide(R, tf.reduce_sum(R, 0))
	return R

def report(training_data, training_target, testing_data, testing_target, k, name):
	R = K_NN(training_data, testing_data, k)
	predictions = tf.matmul(tf.transpose(R), training_target)
	error = tf.divide(tf.reduce_sum(tf.square(predictions - testing_target)), 2*testing_target.shape[0])
	print(name, " error: ", session.run(error))
	return predictions

def get_subplot_index(k):
	map = {}
	map[1] = 1
	map[3] = 2
	map[5] = 3
	map[50] = 4
	return map[k]


# generate test data
np.random.seed(521)
Data = np.linspace(1.0 , 10.0 , num =100) [:, np.newaxis]
Target = np.sin( Data ) + 0.1 * np.power( Data , 2) + 0.5 * np.random.randn(100 , 1)
randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget = Data[randIdx[:80]], Target[randIdx[:80]]
validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]


test_fig = plt.figure()
valid_fig = plt.figure()

session = tf.Session()
for k in {1, 3, 5, 50}:
	print("==========================")
	print("=        k = ", k)
	print("==========================")
	# plot training result
	report(trainData, trainTarget, trainData, trainTarget, k, "training")
	# plot test result
	test_result = report(trainData, trainTarget, testData, testTarget, k, "test")
	test_plot = test_fig.add_subplot(2, 2, get_subplot_index(k))
	test_plot.plot(testData, testTarget, 'g^', testData, session.run(test_result), 'r*', trainData, trainTarget, 'b+')
	test_plot.set_title('Test k = ' + str(k))
	# plot validation result
	validate_result = report(trainData, trainTarget, validData, validTarget, k, "validation")
	valid_plot = valid_fig.add_subplot(2, 2, get_subplot_index(k))
	valid_plot.plot(validData, validTarget, 'g^', validData, session.run(validate_result), 'r*', trainData, trainTarget, 'b+')
	valid_plot.set_title('Validate k = ' + str(k))
	valid_fig.add_subplot(valid_plot)

plt.show()
