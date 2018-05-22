import tensorflow as tf

# L2
# x = tf.placeholder(tf.float32, [None, 2])
# y = tf.placeholder(tf.float32, [None, 1])
# w = tf.truncated_normal([2, 1], stddev=0.1)
# b = tf.Variable(0.)

# h = x * w + b

# y1 = [[10.], [20.]]
# x1 = [[5., 2.], [10., 2.]]

# # 损失函数
# j = 0.1/2 * tf.matmul(tf.transpose(w), w) + 1/2 * tf.reduce_mean(tf.square(y - h))
# # 优化函数
# j_y = tf.train.AdamOptimizer(1e-4).minimize(j)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(10):
#         _, weight, J = sess.run([j_y, w, j], feed_dict={y: y1, x: x1})
#         print(weight)
#         print(J)


# L2
x = tf.placeholder(tf.float32, [None, 2])
y = tf.placeholder(tf.float32, [None, 1])
w = tf.truncated_normal([2, 1], stddev=0.1)
b = tf.Variable(0.)

h = x * w + b

y1 = [[10.], [20.]]
x1 = [[5., 2.], [10., 2.]]

# 损失函数
regularizer = tf.contrib.layers.l2_regularizer(0.1)(w)
j = tf.contrib.layers.apply_regularization(regularizer, weights_list=w) + 1/2 * tf.reduce_mean(tf.square(y - h))

# j = 1/2 * tf.reduce_mean(tf.square(y - h))
# 优化函数
j_y = tf.train.AdamOptimizer(1e-4).minimize(j)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        _, weight, J = sess.run([j_y, w, j], feed_dict={y: y1, x: x1})
        print(weight)
        print(J)
        


# L1
# x = tf.placeholder(tf.float32, [None, 2])
# y = tf.placeholder(tf.float32, [None, 1])
# w = tf.truncated_normal([2, 1], stddev=0.1)
# b = tf.Variable(0.)

# h = x * w + b

# y1 = [[10.], [20.]]
# x1 = [[5., 2.], [10., 2.]]

# # 损失函数
# j = 0.1/2 * tf.reduce_sum(tf.abs(w)) + 1/2 * tf.reduce_mean(tf.square(y - h))
# # 优化函数
# j_y = tf.train.AdamOptimizer(1e-4).minimize(j)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(10):
#         _, weight, J = sess.run([j_y, w, j], feed_dict={y: y1, x: x1})
#         print(weight)
#         print(J)
 