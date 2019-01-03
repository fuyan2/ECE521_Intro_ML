from common import *


def accuracy (weight, bias, input_data, target):
	# predictions for target
	predictions = tf.matmul(input_data, weight) + bias

	# assign classification with 1 or 0
	cond = tf.greater_equal(predictions, tf.constant(0.5, tf.float64))
	predictions = tf.where(cond, tf.ones(cond.shape, tf.uint8), tf.zeros(cond.shape, tf.uint8))

	# calculate prediction accuracy
	_, accuracy = tf.metrics.accuracy(target, predictions)
	tf.local_variables_initializer().run(session=sess)
	return sess.run(accuracy)


def mse(weight, bias, data, target):
	lin_estimations = tf.matmul(data, weight)
	mean_square_error = 0.5 * tf.reduce_mean(tf.square(lin_estimations - tf.cast(target, tf.float64)))
	return sess.run(mean_square_error)


# the training time is significantly lower than SGD
# the training loss is lower than SGD
# however, the validation and test accuracy is about 5%-8% lower
trainData_with_bias = tf.concat([tf.ones(trainTarget.shape, tf.float64), trainData], 1)
validData_with_bias = tf.concat([tf.ones(validTarget.shape, tf.float64), validData], 1)
testData_with_bias = tf.concat([tf.ones(testTarget.shape, tf.float64), testData], 1)
weight = tf.matrix_solve_ls(trainData_with_bias, tf.cast(trainTarget, tf.float64))
bias = 0
trainLoss = 0.5 * tf.reduce_mean(tf.square(tf.matmul(trainData_with_bias, weight) - tf.cast(trainTarget, tf.float64)))
print("trainLoss = ", sess.run(trainLoss))
print("train_accuracy = ", accuracy(weight, bias, trainData_with_bias, trainTarget))
print("valid_accuracy = ", accuracy(weight, bias, validData_with_bias, validTarget))
print("test_accuracy = ", accuracy(weight, bias, testData_with_bias, testTarget))

sess.run(tf.global_variables_initializer())
print("train_mse = ", mse(weight, 0, trainData_with_bias, trainTarget))
print("valid_mse = ", mse(weight, 0, validData_with_bias, validTarget))
print("test_mse = ", mse(weight, 0, testData_with_bias, testTarget))


# linear regression SGD
W, b, loss, accuracy, lin_op = linear_optimizer(0.005, 0)
train_W, train_b, train_losses, _, _, _ = \
	train(lin_op, W, b, loss, accuracy, 500, 20000)

sess = tf.Session()
train_mse = mse(train_W, train_b, trainData, trainTarget)
valid_mse = mse(train_W, train_b, validData, validTarget)
test_mse = mse(train_W, train_b, testData, testTarget)

# compute validation loss
train_accuracy = get_lin_accuracy(train_W, train_b, trainData, trainTarget)
valid_accuracy = get_lin_accuracy(train_W, train_b, validData, validTarget)
test_accuracy = get_lin_accuracy(train_W, train_b, testData, testTarget)

print("linear regression with SGD = ", train_mse)
print("train_mse = ", train_mse)
print("valid_mse = ", valid_mse)
print("test_mse = ", test_mse)

print("Training accuracy = ", train_accuracy)
print("Validation accuracy = ", valid_accuracy)
print("Test accuracy = ", test_accuracy)