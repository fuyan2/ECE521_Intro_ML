import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

with np.load("notMNIST.npz") as data:
	Data, Target = data ["images"], data["labels"]
	np.random.seed(521)
	randIndx = np.arange(len(Data))
	np.random.shuffle(randIndx)
	Data = Data[randIndx]/255.
	Target = Target[randIndx]
	trainData, trainTarget = Data[:15000], Target[:15000]
	validData, validTarget = Data[15000:16000], Target[15000:16000]
	testData, testTarget = Data[16000:], Target[16000:]

trainData = np.reshape(trainData, [trainData.shape[0], -1])
validData = np.reshape(validData, [validData.shape[0], -1])
testData = np.reshape(testData, [testData.shape[0], -1])

trainTarget = np.reshape(trainTarget, [trainTarget.shape[0], -1])
validTarget = np.reshape(validTarget, [validTarget.shape[0], -1])
testTarget = np.reshape(testTarget, [testTarget.shape[0], -1])

# trainData.shape = (15000, 784)
# trainTarget.shape = (15000,)
# validData.shape = (1000, 784)
# validTarget.shape = (1000,)
# testData.shape = (2724, 784)

# parametes
num_class = 10
img_size = 28*28
train_steps = 1000
weight_decay_coeff = 0.0001
learning_rate = 0.0003
num_neurons_each_layer = 500

def plot(name, axis, history):
	axis.plot(range(len(history)), history, label=name)

def layer(input, num_hidden_nodes):
	# input.shape = (batch_size, 784)
	d = input.shape[1].value
	# Xaiver initialization of weight variance
	var = 2.0 / (d + num_hidden_nodes)
	rand = tf.truncated_normal(shape=[d, num_hidden_nodes], \
							   stddev=np.sqrt(var), dtype=tf.float32)
	W = tf.Variable(rand, name='weights')
	b = tf.Variable(tf.zeros([1, num_hidden_nodes], tf.float32), name='bias')
	output = tf.matmul(input, W) + b
	return W, b, output


x = tf.placeholder(tf.float32, [None, img_size], name='X')
y = tf.placeholder(tf.int32, [None, 1], name='y')

W_h1, b_h1, hidden_sum1 = layer(x, num_neurons_each_layer)
hidden1 = tf.nn.relu(hidden_sum1)
# hidden1.shape = (?, 500)
W_h2, b_h2, hidden_sum2 = layer(x, num_neurons_each_layer)
hidden2 = tf.nn.relu(hidden_sum2)
# hidden2.shape = (?, 500)
W_o, b_o, output_sum = layer(hidden2, num_class)
output = tf.nn.relu(output_sum)
# output.shape = (?, 10)

onehot_y = tf.one_hot(tf.reshape(y, shape=[-1]), 10)
# onehot_y.shape = (?, 10)

# softmax predictions for likelihood
softmax_estimations = tf.nn.softmax(output)
# softmax_estimations.shape = (?, 10)

# assign classification with 1 or 0
softmax_predictions = tf.cast(tf.argmax(softmax_estimations, 1), tf.uint8)
# softmax_predictions.shape = (?, )

# calculate prediction accuracy
correct = tf.equal(tf.argmax(softmax_estimations, 1), tf.argmax(onehot_y, 1))
# correct.shape = (?, )
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# cross entropy loss
entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_y, logits=output)
entropy = tf.reduce_mean(entropy)

# weight decay loss
weight_decay_loss = 0.5 * weight_decay_coeff * (tf.reduce_sum(tf.square(W_h1)) + tf.reduce_sum(tf.square(b_h1)) + \
						tf.reduce_sum(tf.square(W_h2)) + tf.reduce_sum(tf.square(b_h2)) + \
						tf.reduce_sum(tf.square(W_o))) + tf.reduce_sum(tf.square(b_o))
# total loss
loss = entropy + weight_decay_loss

# adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
	sess.run(tf.global_variables_initializer())
	_, axis_1 = plt.subplots()
	_, axis_2 = plt.subplots()

	trainLosses = []

	trainErrors = []
	validErrors = []
	testErrors = []

	bestValidAccuracy = 0.0

	for i in range(train_steps):
		_, l = sess.run([optimizer, loss], feed_dict={x: trainData, y: trainTarget})
		trainLosses.append(l)
		print('Training loss at epoch %i: %f' % (i, l))

		trainAccuracy = sess.run(accuracy, feed_dict={x: trainData, y: trainTarget})
		trainErrors.append(1.0 - trainAccuracy)
		
		validAccuracy = sess.run(accuracy, feed_dict={x: validData, y: validTarget})
		validErrors.append(1.0 - validAccuracy)
		if validAccuracy > bestValidAccuracy:
			bestValidAccuracy = validAccuracy

		testAccuracy = sess.run(accuracy, feed_dict={x: testData, y: testTarget})
		testErrors.append(1.0 - testAccuracy)

	plot("Training loss", axis_2, trainLosses)

	plot("Training error", axis_1, trainErrors)
	plot("Validation error", axis_1, validErrors)
	plot("Testing error", axis_1, testErrors)

	print("Best validation error = ", 1.0 - bestValidAccuracy)
	print("Test error = ", testErrors[-1])

	axis_1.legend(loc='upper right')
	axis_2.legend(loc='upper right')
	plt.show()