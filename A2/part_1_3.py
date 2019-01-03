from common import *


# part 1.3
# large weight decay coefficient results in high training mse
# the best weight decay coefficient for validation accuracy is 0.1 or 1
weight_decay_coeffs = [0, 0.001, 0.1, 1]
for i in range(len(weight_decay_coeffs)):
	coeff = weight_decay_coeffs[i]
	W, b, loss, accuracy, lin_op = linear_optimizer(0.005, coeff)
	train_W, train_b, _, _, _, _ = \
	train(lin_op, W, b, loss, accuracy, 500, iterations)

	# compute validation loss
	valid_accuracy = get_lin_accuracy(train_W, train_b, validData, validTarget)
	# compute test loss
	test_accuracy = get_lin_accuracy(train_W, train_b, testData, testTarget)

	print("weight decay coeff = ", coeff)
	print("Validation accuracy = ", valid_accuracy)
	print("Test accuracy = ", test_accuracy)