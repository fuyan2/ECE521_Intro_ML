from data import *
import numpy as np

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

import tensorflow as tf

# parametes
num_class = 10
img_size = 28*28
train_steps = 500
weight_decay_coeff = 0.0001
learning_rate = 0.0003
num_neurons_each_layer = 1000

def layer(input, num_hidden_nodes):
	# input.shape = (batch_size, 784)
	d = input.shape[1].value
	# Xaiver initialization of weight variance
	var = 3.0 / (d + num_hidden_nodes)
	rand = tf.truncated_normal(shape=[d, num_hidden_nodes], \
							   stddev=np.sqrt(var), dtype=tf.float32)
	W = tf.Variable(rand, name='weights')
	b = tf.Variable(tf.zeros([1, num_hidden_nodes], tf.float32), name='bias')
	output = tf.matmul(input, W) + b
	return W, b, output


x = tf.placeholder(tf.float32, [None, img_size], name='X')
y = tf.placeholder(tf.int32, [None, 1], name='y')

W_h, b_h, hidden_sum = layer(x, num_neurons_each_layer)
hidden = tf.nn.relu(hidden_sum)
# onehot_y.shape = (?, 1000)
W_o, b_o, output_sum = layer(hidden, num_class)
output = tf.nn.relu(output_sum)
# onehot_y.shape = (?, 10)

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
weight_decay_loss = 0.5 * weight_decay_coeff * (tf.reduce_sum(tf.square(W_h)) + tf.reduce_sum(tf.square(b_h)) + \
						tf.reduce_sum(tf.square(W_o))) + tf.reduce_sum(tf.square(b_o))
# total loss
loss = entropy + weight_decay_loss

# adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)