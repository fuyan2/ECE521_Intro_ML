from common import *


# part 1.1
# the best learning rate seems to be 0.005
_, axis_1 = plt.subplots()

rates = [0.005, 0.001, 0.0001]
for i in range(len(rates)):
	rate = rates[i]
	W, b, loss, accuracy, lin_op = linear_optimizer(rate, 0)
	train_W, train_b, train_losses, train_accuracies, valid_losses, valid_accuracies = \
		train(lin_op, W, b, loss, accuracy, 500, iterations)

	# compute validation loss
	valid_accuracy = get_lin_accuracy(train_W, train_b, validData, validTarget)

	plot("learning rate = " + str(rate), axis_1, train_losses)
	print("rate = ", rate)
	print("Training loss = ", train_losses[-1])
	print("Validation accuracy = ", valid_accuracy)

axis_1.legend(loc='upper right')
plt.show()