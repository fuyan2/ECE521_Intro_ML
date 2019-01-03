from common import *

samples = 2000
_, axis_1 = plt.subplots()

mse = [0]
entropy = [0]
y_pred = [0]

for i in range(1, samples):
    x = float(i) / samples
    y_pred.append(x)
    mse.append(0.5*x*x)

y_label = tf.zeros_like(mse)
cross = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_label, logits=y_pred)
entropy = sess.run(cross)

axis_1.plot(range(len(mse)), mse, label="Mean Square Error")
axis_1.plot(range(len(entropy)), entropy, label="Cross Entropy Loss")

axis_1.legend(loc='lower right')
plt.show()