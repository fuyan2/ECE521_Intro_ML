import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from common import *

# part 2.1.1
# step 1 get best learning rate
# the best learning rate seems to be 0.005
rates = [0.005, 0.001, 0.0001]
for i in range(len(rates)):
	rate = rates[i]
	W, b, loss, accuracy, log_op = logistic_optimizer(rate, 0.01)
	train_W, train_b, train_losses, train_accuracies, valid_losses, valid_accuracies = \
		train(log_op, W, b, loss, accuracy, 500, 5000)

	# compute validation loss
	valid_accuracy = get_log_accuracy(train_W, train_b, validData, validTarget)

	print("rate = ", rate)
	print("Training loss = ", train_losses[-1])
	print("Validation accuracy = ", valid_accuracy)