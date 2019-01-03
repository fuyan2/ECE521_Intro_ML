import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from common import *

# part 2.1.3
# comparing linear and logistic regression
# linear regression is faster when using normal equation,
# but training, validation, and test accuracy are all lower
# than logistic regression

_, axis_1 = plt.subplots()
_, axis_2 = plt.subplots()

rate = 0.001

#  logistic regression part
# set weight decay to be zero
W, b, loss, accuracy, adam_op = log_adam_optimizer(rate, 0)
train_W, train_b, train_losses, train_accuracies, valid_losses, valid_accuracies = \
	train(adam_op, W, b, loss, accuracy, 500, 5000)

plot("Training losses of logistic regression", axis_1, train_losses)
plot("Training accuracies of logistic regression", axis_2, train_accuracies)

# compute accuracies
train_accuracy = get_log_accuracy(train_W, train_b, trainData, trainTarget)
valid_accuracy = get_log_accuracy(train_W, train_b, validData, validTarget)
test_accuracy = get_log_accuracy(train_W, train_b, testData, testTarget)

print("Logistic Regression")
print("Training acaccuracy = ", train_accuracy)
print("Validation accuracy = ", valid_accuracy)
print("Test accuracy = ", test_accuracy)



#  linear regression part
# set weight decay to be zero
W, b, loss, accuracy, lin_adam_op = lin_adam_optimizer(rate, 0)
train_W, train_b, train_losses, train_accuracies, valid_losses, valid_accuracies = \
	train(lin_adam_op, W, b, loss, accuracy, 500, 5000)

plot("Training loss of linear regression", axis_1, train_losses)
plot("Training accuracy of linear regression", axis_2, train_accuracies)

# compute accuracies
train_accuracy = get_lin_accuracy(train_W, train_b, trainData, trainTarget)
valid_accuracy = get_lin_accuracy(train_W, train_b, validData, validTarget)
test_accuracy = get_lin_accuracy(train_W, train_b, testData, testTarget)

print("Linear Regression")
print("Training acaccuracy = ", train_accuracy)
print("Validation accuracy = ", valid_accuracy)
print("Test accuracy = ", test_accuracy)


axis_1.legend(loc='upper right')
axis_2.legend(loc='lower right')
plt.show()