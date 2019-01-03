from common import *


# part 1.2
# as batch_size increase to the size of training data,
# the training MSE reduces, however, the training time increases
_, axis_1 = plt.subplots()

batch_sizes = [500, 1500, 3500]
for i in range(len(batch_sizes)):
	W, b, loss, accuracy, lin_op = linear_optimizer(0.005, 0)
	batch_size = batch_sizes[i]
	train_W, train_b, train_losses, _, _, _ = \
	train(lin_op, W, b, loss, accuracy, batch_size, iterations)

	# compute validation loss
	valid_accuracy = get_lin_accuracy(train_W, train_b, validData, validTarget)

	plot("batch size = " + str(batch_size), axis_1, train_losses)
	print("batch_size = ", batch_size)
	print("Training loss = ", train_losses[-1])
	print("Validation accuracy = ", valid_accuracy)

axis_1.legend(loc='upper right')
plt.show()