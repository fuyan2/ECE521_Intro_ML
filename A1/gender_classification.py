import numpy as np
import tensorflow as tf

# calculate euclidean distance square
def D_eucl_sqr(X, Z):
	if X.shape[1] != Z.shape[1]:
		print("D_eucl_sqr: X and Z must have the same number of column")
		return tf.constant([])
	return tf.reduce_sum(tf.square(X[:, None] - Z[None, :]), 2)

# find k nearest neighbors, return responsility matrix
def K_NN(X, Z, k):
	if X.shape[0] < k:
		print("k exceeds the maximum allowed number by matrix!")
		return None;
	distance = D_eucl_sqr(Z, X)
	reciprocal = tf.reciprocal(distance)
	values, indices = tf.nn.top_k(reciprocal, k)
	return indices

def data_segmentation(data_path, target_path, task):
	# task = 0 >> select the name ID targets for face recognition task
	# task = 1 >> select the gender ID targets for gender recognition task
	data = np.load(data_path)/255
	data = np.reshape(data, [-1, 32*32])

	target = np.load(target_path)
	np.random.seed(45689)
	rnd_idx = np.arange(np.shape(data)[0])
	np.random.shuffle(rnd_idx)
	trBatch = int(0.8*len(rnd_idx))
	validBatch = int(0.1*len(rnd_idx))

	trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
	data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
	data[rnd_idx[trBatch + validBatch+1:-1],:]

	trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
	target[rnd_idx[trBatch+1:trBatch + validBatch], task],\
	target[rnd_idx[trBatch + validBatch + 1:-1], task]

	return trainData, validData, testData, trainTarget, validTarget, testTarget


session = tf.Session()
trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation("data.npy", "target.npy", 1)

for k in {1, 5, 10, 25, 50, 100, 200}:
	kVals = K_NN(trainData, testData, k)
	votes = tf.cast(tf.gather(trainTarget, kVals), tf.int64)
	predictions = tf.map_fn(lambda v: tf.gather(tf.unique_with_counts(v)[0], tf.argmax(tf.unique_with_counts(v)[2])), votes)

	err = tf.equal(predictions, testTarget)[:, None]
	indicator = tf.where(err, tf.zeros(err.shape, tf.float64), tf.ones(err.shape, tf.float64))
	error = tf.reduce_mean(indicator)
	print(k, "test error: ", session.run(error))

	kVals = K_NN(trainData, validData, k)
	votes = tf.cast(tf.gather(trainTarget, kVals), tf.int64)
	predictions = tf.map_fn(lambda v: tf.gather(tf.unique_with_counts(v)[0], tf.argmax(tf.unique_with_counts(v)[2])), votes)

	err = tf.equal(predictions, validTarget)[:, None]
	indicator = tf.where(err, tf.zeros(err.shape, tf.float64), tf.ones(err.shape, tf.float64))
	error = tf.reduce_mean(indicator)
	print(k, "validate error: ", session.run(error))