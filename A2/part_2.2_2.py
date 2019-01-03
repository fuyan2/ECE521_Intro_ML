import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

# task = 0 >> select the name ID targets for face recognition task
# task = 1 >> select the gender ID targets for gender recognition task
data = np.load("../A1/data.npy")/255
data = np.reshape(data, [-1, 32*32])

target = np.load("../A1/target.npy")
np.random.seed(45689)
rnd_idx = np.arange(np.shape(data)[0])
np.random.shuffle(rnd_idx)
trBatch = int(0.8*len(rnd_idx))
validBatch = int(0.1*len(rnd_idx))

trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
data[rnd_idx[trBatch + validBatch+1:-1],:]

trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], 0], \
target[rnd_idx[trBatch+1:trBatch + validBatch], 0],\
target[rnd_idx[trBatch + validBatch + 1:-1], 0]

# flatten the data to rank 2
trainData = np.reshape(trainData, [trainData.shape[0], -1])
validData = np.reshape(validData, [validData.shape[0], -1])
testData = np.reshape(testData, [testData.shape[0], -1])

trainTarget = np.reshape(trainTarget, [trainTarget.shape[0], -1])
validTarget = np.reshape(validTarget, [validTarget.shape[0], -1])
testTarget = np.reshape(testTarget, [testTarget.shape[0], -1])


# training inputs
X = tf.placeholder(tf.float64, [None, 1024], name='input_x')
y_target = tf.placeholder(tf.uint8, [None, 1], name='target_y')
y = tf.cast(y_target, tf.float64)

# softmax labels: each row is a probability distribution
# in this case, each row will have only one field equals to 1
multi_y = tf.one_hot(tf.reshape(y_target, shape=[-1]), 10)

# trainable variables
rand = tf.truncated_normal(shape=[1024, 10], stddev=0.5, dtype=tf.float64)
W = tf.Variable(rand, name='weights')
b = tf.Variable(tf.zeros(1, tf.float64), name='bias')


# softmax predictions for likelihood
lin_estimations = tf.matmul(X, W) + b
softmax_estimations = tf.nn.softmax(lin_estimations)

# assign classification with 1 or 0
softmax_predictions = tf.cast(tf.argmax(softmax_estimations, 1), tf.uint8)


# calculate prediction accuracy
correct = tf.equal(tf.argmax(softmax_estimations, 1), tf.argmax(multi_y, 1))
softmax_accuracy = tf.reduce_mean(tf.cast(correct, tf.float64))


def plot(name, axis, history):
	axis.plot(range(len(history)), history, label=name)

def softmax_adam_optimizer(learning_rate, weight_decay_coeff):
	# cross entropy loss
	entropy = tf.nn.softmax_cross_entropy_with_logits(labels=multi_y, logits=lin_estimations)
	entropy = tf.reduce_mean(entropy)

	# weight decay loss
	weight_decay_loss = 0.5 * weight_decay_coeff * tf.reduce_sum(tf.square(W))
	# total loss
	loss = entropy + weight_decay_loss

	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	return W, b, loss, softmax_accuracy, optimizer


def train_multi(optimizer, W, b, loss, accuracy, batch_size, iterations):
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


def get_softmax_accuracy (weight, bias, input_data, target):
	return sess.run(softmax_accuracy, feed_dict={X: input_data, y_target: target})

# tune learning_rate and weight decay coefficient
# rates = [0.005, 0.001, 0.0001]
# for i in range(len(rates)):
# 	learning_rate = rates[i]

# 	weight_decay_coeffs = [0, 0.001, 0.1, 1]
# 	for i in range(len(weight_decay_coeffs)):
# 		coeff = weight_decay_coeffs[i]
# 		W, b, loss, accuracy, softmax_op = \
# 			softmax_adam_optimizer(learning_rate, coeff)

# 		train_W, train_b, train_losses, train_accuracies, valid_losses, valid_accuracies = \
# 			train_multi(softmax_op, W, b, loss, accuracy, 300, 10000)

# 		print("=====================================")
# 		print("learning rate = ", learning_rate)
# 		print("weight decay coeff = ", coeff)
# 		print("Training loss = ", train_losses[-1])
# 		print("Validation accuracy = ", get_softmax_accuracy(train_W, train_b, validData, validTarget))
# 		print("Test accuracy = ", get_softmax_accuracy(train_W, train_b, testData, testTarget))


_, axis_1 = plt.subplots()
_, axis_2 = plt.subplots()

learning_rate = 0.005
coeff = 0.01
W, b, loss, accuracy, softmax_op = \
	softmax_adam_optimizer(learning_rate, coeff)

train_W, train_b, train_losses, train_accuracies, valid_losses, valid_accuracies = \
	train_multi(softmax_op, W, b, loss, accuracy, 300, 10000)

print("=====================================")
print("learning rate = ", learning_rate)
print("weight decay coeff = ", coeff)
print("Training loss = ", train_losses[-1])
print("Validation accuracy = ", get_softmax_accuracy(train_W, train_b, validData, validTarget))
print("Test accuracy = ", get_softmax_accuracy(train_W, train_b, testData, testTarget))

plot("Train losses", axis_1, train_losses)
plot("Train accuracies", axis_2, train_accuracies)
plot("Valid losses", axis_1, valid_losses)
plot("Valid accuracies", axis_2, valid_accuracies)


axis_1.legend(loc='upper right')
axis_2.legend(loc='lower right')
plt.show()