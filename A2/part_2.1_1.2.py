import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from common import *

# part 2.1.1
# step 2 get plot
# the best learning rate seems to be 0.005
_, axis_1 = plt.subplots()
_, axis_2 = plt.subplots()

rate = 0.005
W, b, loss, accuracy, log_op = logistic_optimizer(rate, 0.01)
train_W, train_b, train_losses, train_accuracies, valid_losses, valid_accuracies = \
	train(log_op, W, b, loss, accuracy, 500, 5000)

# compute validation loss
valid_accuracy = get_log_accuracy(train_W, train_b, validData, validTarget)
test_accuracy = get_log_accuracy(train_W, train_b, testData, testTarget)

plot("Training loss", axis_1, train_losses)
plot("Training accuracy", axis_2, train_accuracies)
plot("Validation loss", axis_1, valid_losses)
plot("Validation accuracy", axis_2, valid_accuracies)

print("Training loss = ", train_losses[-1])
print("Training acaccuracy = ", train_accuracies[-1])
print("Validation accuracy = ", valid_accuracy)
print("Test accuracy = ", test_accuracy)

axis_1.legend(loc='upper right')
axis_2.legend(loc='lower right')
plt.show()