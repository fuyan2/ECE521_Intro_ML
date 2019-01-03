import tensorflow as tf


# calculate euclidean distance square
def D_eucl_sqr(X, Z):
	if X.shape[1] != Z.shape[1]:
		print("D_eucl_sqr: X and Z must have the same number of column")
		return tf.constant([])
	return tf.reduce_sum(tf.square(X[:, None] - Z[None, :]), 2)


session = tf.Session()

X = tf.constant([[1,2,2,1,2], [4,2,3,2,1], [2,3,3,2,5]], name='x', dtype=tf.float32)
Z = tf.constant([[1,2,1,4,2], [2,1,4,3,2]], name='z', dtype=tf.float32)

D = D_eucl(X, Z)
print(session.run(D))