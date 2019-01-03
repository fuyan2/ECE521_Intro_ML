import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

iterations = 20000
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

with np.load("notMNIST.npz") as data:
	Data, Target = data["images"], data["labels"]
	posClass = 2
	negClass = 9
	dataIndx = (Target==posClass) + (Target==negClass)
	Data = Data[dataIndx]/255.
	Target = Target[dataIndx].reshape(-1, 1)
	Target[Target==posClass] = 1
	Target[Target==negClass] = 0
	np.random.seed(521)
	randIndx = np.arange(len(Data))
	np.random.shuffle(randIndx)
	Data, Target = Data[randIndx], Target[randIndx]
	trainData, trainTarget = Data[:3500], Target[:3500]
	validData, validTarget = Data[3500:3600], Target[3500:3600]
	testData, testTarget = Data[3600:], Target[3600:]

# flatten the data to rank 2
trainData = np.reshape(trainData, [trainData.shape[0], -1])
validData = np.reshape(validData, [validData.shape[0], -1])
testData = np.reshape(testData, [testData.shape[0], -1])


# training inputs
X = tf.placeholder(tf.float64, [None, 784], name='input_x')
y_target = tf.placeholder(tf.uint8, [None, 1], name='target_y')
y = tf.cast(y_target, tf.float64)

# trainable variables
rand = tf.truncated_normal(shape=[784, 1], stddev=0.5, dtype=tf.float64)
W = tf.Variable(rand, name='weights')
b = tf.Variable(tf.zeros(1, tf.float64), name='bias')


# for loss and accurarcy
# linear predictions for target
lin_estimations = tf.matmul(X, W) + b

# assign classification with 1 or 0
cond = tf.greater_equal(lin_estimations, tf.constant(0.5, tf.float64))
lin_predictions = tf.where(cond, tf.ones_like(y_target), tf.zeros_like(y_target))

# calculate prediction accuracy
_, lin_accuracy = tf.metrics.accuracy(y_target, lin_predictions)
sess.run(tf.local_variables_initializer())

# logistic predictions for likelihood
log_estimations = tf.sigmoid(tf.matmul(X, W) + b)

# assign classification with 1 or 0
cond = tf.greater_equal(log_estimations, tf.constant(0.5, tf.float64))
log_predictions = tf.where(cond, tf.ones_like(y_target), tf.zeros_like(y_target))

# calculate prediction accuracy
_, log_accuracy = tf.metrics.accuracy(y_target, log_predictions)
sess.run(tf.local_variables_initializer())


def plot(name, axis, history):
	axis.plot(range(len(history)), history, label=name)


def linear_optimizer(learning_rate, weight_decay_coeff):
	# mean_square error loss
	mean_square_error = 0.5 * tf.reduce_mean(tf.square(lin_estimations - y))

	# weight decay loss
	weight_decay_loss = 0.5 * weight_decay_coeff * tf.reduce_sum(tf.square(W))
	# total loss
	loss = mean_square_error + weight_decay_loss

	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	return W, b, loss, lin_accuracy, optimizer


def logistic_optimizer(learning_rate, weight_decay_coeff):
	# predictions for likelihood
	log_predictions = tf.sigmoid(tf.matmul(X, W) + b)

	# cross entropy loss
	entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=log_estimations)
	entropy = tf.reduce_mean(entropy)

	# weight decay loss
	weight_decay_loss = 0.5 * weight_decay_coeff * tf.reduce_sum(tf.square(W))
	# total loss
	loss = entropy + weight_decay_loss

	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	return W, b, loss, log_accuracy, optimizer


def lin_adam_optimizer(learning_rate, weight_decay_coeff):
	# mean_square error loss
	mean_square_error = 0.5 * tf.reduce_mean(tf.square(lin_estimations - y))

	# weight decay loss
	weight_decay_loss = 0.5 * weight_decay_coeff * tf.reduce_sum(tf.square(W))
	# total loss
	loss = mean_square_error + weight_decay_loss

	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	return W, b, loss, lin_accuracy, optimizer


def log_adam_optimizer(learning_rate, weight_decay_coeff):
	# cross entropy loss
	entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=log_estimations)
	entropy = tf.reduce_mean(entropy)

	# weight decay loss
	weight_decay_loss = 0.5 * weight_decay_coeff * tf.reduce_sum(tf.square(W))
	# total loss
	loss = entropy + weight_decay_loss

	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	return W, b, loss, log_accuracy, optimizer


def train(optimizer, W, b, loss, accuracy, batch_size, iterations):
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	batch_start = 0
	N = trainData.shape[0]

	train_losses = []
	train_accuracies = []
	valid_losses = []
	valid_accuracies = []

	for i in range(0, iterations):
		batch_end = batch_start + batch_size
		if (batch_end <= N):
			batch_x = trainData[batch_start:batch_start+batch_size]
			batch_y = trainTarget[batch_start:batch_start+batch_size]
		else:
			# wrap around the array to get sufficient number of samples
			batch_x = np.concatenate((trainData[batch_start:], trainData[:batch_end - N]))
			batch_y = np.concatenate((trainTarget[batch_start:], trainTarget[:batch_end - N]))

		_, currentW, currentb = \
			sess.run([optimizer, W, b], feed_dict={X: batch_x, y_target: batch_y})

		batch_start += batch_size
		if (batch_start >= N):
			# we have covered the entire set
			# calculate training loss and accuracy over entire training set
			train_loss, train_accuracy = \
				sess.run([loss, accuracy], feed_dict={X: trainData, y_target: trainTarget})

			# calculate validation loss and accuracy
			valid_loss, valid_accuracy = \
				sess.run([loss, accuracy], feed_dict={X: validData, y_target: validTarget})

			train_losses.append(train_loss)
			train_accuracies.append(train_accuracy)
			valid_losses.append(valid_loss)
			valid_accuracies.append(valid_accuracy)

			batch_start = batch_start - N
	return currentW, currentb, train_losses, train_accuracies, valid_losses, valid_accuracies


def get_lin_accuracy (weight, bias, input_data, target):
	return sess.run(lin_accuracy, feed_dict={X: input_data, y_target: target})


def get_log_accuracy (weight, bias, input_data, target):
	return sess.run(log_accuracy, feed_dict={X: input_data, y_target: target})


def get_log_loss (weight, bias, input_data, target, weight_decay_coeff=0.01):
	# prediction for likelihood
	predictions = tf.sigmoid(tf.matmul(input_data, weight) + bias)

	labels=tf.cast(target, tf.float64)

	# cross entropy loss
	entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=predictions)
	entropy = tf.reduce_mean(entropy)

	# weight decay loss
	weight_decay_loss = 0.5 * weight_decay_coeff * tf.reduce_sum(tf.square(W))

	# totol loss
	loss = entropy + weight_decay_loss
	return sess.run(loss)