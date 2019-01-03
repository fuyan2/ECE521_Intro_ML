import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from common import *


# part 1.1
# the best learning rate seems to be 0.005
_, axis_1 = plt.subplots()

rates = [0.005, 0.001, 0.0001]
for i in range(len(rates)):
	rate = rates[i]
	W, b, pred, loss, lin_op = linear_optimizer(rate, 0)
	train_W, train_b, train_history = \
		train(lin_op, W, b, loss, trainData, trainTarget, 500, iterations)

	# compute validation loss
	valid_accuracy = get_accuracy(train_W, train_b, validData, validTarget)

	plot("learing rate = " + str(rate), axis_1, train_history[::7])
	print("rate = ", rate)
	print("Training loss = ", train_history[-1])
	print("Validation accuracy = ", valid_accuracy)
axis_1.legend(loc='upper right')


# part 1.2
# as batch_size increase to the size of training data,
# the training MSE reduces, however, the training time increases
_, axis_2 = plt.subplots()

batch_sizes = [500, 1500, 3500]
for i in range(len(batch_sizes)):
	W, b, pred, loss, lin_op = linear_optimizer(0.005, 0)
	batch_size = batch_sizes[i]
	train_W, train_b, train_history = \
	train(lin_op, W, b, loss, trainData, trainTarget, batch_size, iterations)

	# compute validation loss
	valid_accuracy = get_accuracy(train_W, train_b, validData, validTarget)

	plot("batch size = " + str(batch_size), axis_2, train_history)
	print("batch_size = ", batch_size)
	print("Training loss = ", train_history[-1])
	print("Validation accuracy = ", valid_accuracy)
axis_2.legend(loc='upper right')


# # part 1.3
# # large weight decay coefficient results in high training mse
# # the best weight decay coefficient seems to be 0 (or 0.001)
weight_decay_coeffs = [0, 0.001, 0.1, 1]
for i in range(len(weight_decay_coeffs)):
	coeff = weight_decay_coeffs[i]
	W, b, pred, loss, lin_op = linear_optimizer(0.005, coeff)
	train_W, train_b, train_history = \
	train(lin_op, W, b, loss, trainData, trainTarget, 500, iterations)

	# compute validation loss
	valid_accuracy = get_accuracy(train_W, train_b, validData, validTarget)

	print("weight decay coeff = ", coeff)
	print("Validation accuracy = ", valid_accuracy)

plt.show()
