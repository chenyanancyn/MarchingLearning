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


# # L2
# x = tf.constant([[5., 2.], [10., 2.]])   # 不用placeholder,在run时，就不用feed_dict
# y = tf.constant([[10.], [20.]])
# w = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
# b = tf.Variable(0.)

# h = tf.matmul(x,  w) + b

# # 正则化
# tf.add_to_collection(tf.GraphKeys.WEIGHTS, w)  
# l = tf.get_collection(tf.GraphKeys.WEIGHTS)
# regularizer = tf.contrib.layers.l2_regularizer(0.1)
# # 损失函数正则化
# reg_loss = tf.contrib.layers.apply_regularization(regularizer, l)
# j = tf.reduce_mean(tf.square(y - h)) + reg_loss

# # # j = 1/2 * tf.reduce_mean(tf.square(y - h))
# # # 优化函数
# j_y = tf.train.AdamOptimizer(.1).minimize(j)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())

#     for i in range(300):
#         _, weight, J = sess.run([j_y, w, j])
#         if i % 20 == 0:
#             print(weight)
#             print(J)
        


# # L1
# # x = tf.placeholder(tf.float32, [None, 2])
# # y = tf.placeholder(tf.float32, [None, 1])
# # w = tf.truncated_normal([2, 1], stddev=0.1)
# # b = tf.Variable(0.)

# # h = x * w + b

# # y1 = [[10.], [20.]]
# # x1 = [[5., 2.], [10., 2.]]

# # # 损失函数
# # j = 0.1/2 * tf.reduce_sum(tf.abs(w)) + 1/2 * tf.reduce_mean(tf.square(y - h))
# # # 优化函数
# # j_y = tf.train.AdamOptimizer(1e-4).minimize(j)

# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #     for i in range(10):
# #         _, weight, J = sess.run([j_y, w, j], feed_dict={y: y1, x: x1})
# #         print(weight)
# #         print(J)




# # tf.get_collection  # 将一些变量或op放到一个list中，
# 创建模型
# w1 = tf.Variable(1.)
# w2 = tf.Variable(2.)
# tf.add_to_collection(tf.GraphKeys.WEIGHTS, [w1, w2])   # w1 w2加入到集合中
# print(tf.get_collection(tf.GraphKeys.WEIGHTS))


# tf.contrib.slim.conv2d   将weights和bias加入到对应GraphKeys中

# # 声明正则化方法 **
# r = tf.contrib.layers.l2_regularizer()
# tf.add_to_collection(tf.GraphKeys.WEIGHTS, [w1, w2])  
# l = tf.get_collection(tf.GraphKeys.WEIGHTS)
# tf.contrib.layers.apply_regularization(r, l)


# # 正则化在tf.slim中的使用
# r = tf.contrib.layers.l1_l2_regularizer(1.)
# tf.contrib.slim.conv2d(
#     tf.ones([5, 10, 10, 1]),
#     8,
#     [3, 3],
#     weights_regularizer=r
# )

# reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
# print(reg_loss)


# # 生成伯努利分布，并用于隐藏层
# keep_prob = 0.8
# drop_prob = 1 - keep_prob
# rdu = tf.random_uniform([3], 0, 1)
# r = tf.cast(tf.equal(tf.minimum(rdu, drop_prob), drop_prob), tf.float32)
# hidden_output = tf.constant([[1, 2, 3], [4, 5, 6]])
# res = tf.multiply(hidden_output, r)


# 0314
# 优化方法
# import tensorflow as tf

# x = tf.constant([[5., 2.], [10., 2.], [20., 2.]])
# y = tf.constant([[10.], [20.], [40.]])

# w = tf.Variable(tf.random_normal([2, 1]))
# b = tf.Variable(0.)

# h = tf.matmul(x, w) + b

# loss = tf.reduce_mean(tf.square(y - h))    # 代价

# learning_rate = 0.003
# opt = tf.train.GradientDescentOptimizer(learning_rate)   # 优化器
# train_op = opt.minimize(loss)

# # 方法一
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #     for i in range(10):
# #         print(sess.run([loss, train_op])[0])


# # 方法二
# learning_rate = 0.003
# var_list = [w, b]
# opt = tf.train.GradientDescentOptimizer(learning_rate)
# # 计算梯度
# grads_and_vars = opt.compute_gradients(loss, var_list)   # 是对w和b求梯度
# train_op = opt.apply_gradients(grads_and_vars)    # 与上面得到的minimize一样的

# 方法三
# 梯度消失： 增大学习率， 更新激活函数
# 梯度爆炸：梯度裁剪 
# learning_rate = 0.003
# var_list = [w, b]

# # opt = tf.train.Optimizer(use_locking=False, name='cus_op')   # 梯度下降优化器的父类
# grads_vars = opt.compute_gradients(loss, var_list)

# # theta = theta - learning_rate*grad
# train_op = [gv[1].assign_sub(learning_rate*gv[0]) for gv in grads_vars]   # 不等于上面的train_op, 但功能类似
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(10):
#         print(sess.run([loss, train_op])[0])




# 0316
# 动量梯度下降的tf实现
# class MomentumOpt(tf.train.Optimizer):

#     def __init__(self, learning_rate=0.001, momentum=0.9,
#                 use_locking=False, name='CusMom'):
#         super(MomentumOpt, self).__init__(use_locking, name)
#         self._lr = learning_rate
#         self._momentum = momentum

#         self._lr_t = None
#         self._mom_t = None

#     def _prepare(self):
#         self._lr_t = tf.comvert_to_tensor(self._lr)
#         self._mom_t = tf.convert_to_tensor(self._momentum)

#     def _create_slots(self, var_list):
#         for v in var_list:
#             self._zeros_slot(v, 'v', self._name)

#     def _apply_dense(self, grad, var):
#         self._lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
#         self._mom_t = tf.case(self._mom_t, var.dtype.base_dtype)

#         v = self.get_slot(var, 'v')
#         v_t = v.assign(v*self._mom_t - grad*self._lr_t)
#         var_update = var.assign_add(v_t)

#         return var_update


# 对比
# import tensorflow as tf

# x = tf.constant([[5., 2.], [10., 2.], [20., 2.]])
# y = tf.constant([[10.], [20.], [40.]])

# w = tf.Variable([[2.], [1.]])
# b = tf.Variable(0.)

# h = tf.matmul(x, w) + b

# loss = tf.reduce_mean(tf.square(y - h))    # 代价

# # 下降算法一
# learning_rate = 0.003
# opt = tf.train.GradientDescentOptimizer(learning_rate)   # 优化器
# train_op = opt.minimize(loss)


# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(10):
#         print(sess.run([loss, train_op])[0])


# import tensorflow as tf

# x = tf.constant([[5., 2.], [10., 2.], [20., 2.]])
# y = tf.constant([[10.], [20.], [40.]])   

# w = tf.Variable([[2.], [1.]])
# b = tf.Variable(0.)

# h = tf.matmul(x, w) + b

# loss = tf.reduce_mean(tf.square(y - h))    # 代价

# # 下降算法二
# learning_rate = 0.003
# momentum = 0.9
# opt = tf.train.MomentumOptimizer(learning_rate, momentum)   # 优化器
# train_op = opt.minimize(loss)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(10):
#         print(sess.run([loss, train_op])[0])

# nesterov
# import tensorflow as tf

# x = tf.constant([[5., 2.], [10., 2.], [20., 2.]])
# y = tf.constant([[10.], [20.], [40.]])   

# w = tf.Variable([[2.], [1.]])
# b = tf.Variable(0.)

# h = tf.matmul(x, w) + b

# loss = tf.reduce_mean(tf.square(y - h))    # 代价

# # 下降算法二
# learning_rate = 0.003
# momentum = 0.9
# opt = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)   # 优化器
# train_op = opt.minimize(loss)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(10000):
#         if i < 10:
#             loss_res, _ = sess.run([loss, train_op])
#             print(loss_res)
#         if loss_res < 1e-5:
#             print(i)
#             break


# # RMSProp 算法实现
# class RMSProp(tf.train.Optimizer):
#     def __init__(self, learning_rate=0.001, decay=0.9, epsilon=1e-6,
#                 use_locking=False, name='CusRMSProp'):
#         super(RMSProp, self).__init__(use_locking, name)
#         self._lr = learning_rate
#         self._decay = decay
#         self._ep = epsilon

#         self._lr_t = None
#         self._decay_t = None
#         self._ep_t = None

#     def _prepare(self):
#         self._lr_t = tf.convert_to_tensor(self._lr)
#         self._decay_t = tf.convert_tensor(self._decay)
#         self._ep_t = tf.convert_to_tensor(self._ep)

#     def _create_slots(self, var_list):   # 在执行minmize中使用
#         for r in var_list:
#             self._zeros_slot(r, 'r', self._name)  # 'r' 表示的一个集合
    
#     def _apply_dense(self, grad, var):   # 在执行minmize中使用
#         r = self.get_slot(var, 'r')
#         update_r = r.assign(r*self._decay_t+(1-self._decay_t)*grad*grad)
#         delta_theta = -(self._lr_t*grad)/(tf.sqrt(updata_r)+self._ep_t)
#         update_var = var.assign_add(delta_theta)
#         return update_var


# # Adam实现
# class Adam(tf.train.Optimizer):
#     def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999
#                 epsilon=1e-08, use_locking=False, name='Adam'):
#         super(Adam, self).__init__(use_lockiing, name)
#         self._lr = learning_rate
#         self._b1 = beta1
#         self._b2 = beta2
#         self._ep = epsilon

#         self._lr_t = None
#         self._b1_t = None
#         self._b2_t = None
#         self._ep_t = None
#         self._t = None

#     def _prepare(self):
#         self._lr_t = tf.convert_to_tensor(self._lr)
#         self._b1_t = tf.convert_to_tensor(self._b1)
#         self._b2_t = tf.convert_to_tensor(self._b2)
#         self._ep_t = tf.convert_to_tensor(self._ep)
#         self._t = tf.Variable(0.)

#     def _create_slots(self, var_list):
#         for var in var_list:
#             self._zeros_slot(var, 's', self._name)
#             self._zeros_slot(var, 'r', self._name)

#     def _apply_dense(self, grad, var):
#         update_t = self._t.assign_add(1)
#         s = self.get_slot(var, 's')
#         update_s = s.assign(s*self._b1_t+grad*(1-self._b1_t))
#         r = self.get_slot(var, 'r')
#         update_r = r.assign(r*self._b2_t+grad*grad*(1-self.b2_t))
#         fix_s = update_s / (1-tf.pow(self._b1_t, update_t))
#         fix_r = update_r / (1-tf.pow(self._b2_t, update_t))
#         delta_theta = -(self._lr_t*fix_s)/(self._ep_t+tf.sqrt(fix_r))
#         update_var = var.assign_add(delta_theta)

#         return update_var



# # AlexNet

# num_classes = 1000
# inputs_img = tf.palceholder(shape=[None, 227, 227, 3], dtype=tf.float32)
# labels = tf.placeholder(shape=[None, num_classes], dtype=tf.int32)

# net_GPU_0 = slim.conv2d(inputs_img, 48, [11, 11], stride=4, padding='VALID')
# net_GPU_1 = slim.conv2d(inputs_img, 48, [11, 11], stride=4, padding='VALID')

# net_GPU_0 = slim.max_pool2d(net_GPU_0, [3, 3], stride=2, padding='VALID')
# net_GPU_1 = slim.max_pool2d(net_GPU_1, [3, 3], stride=2, padding='VALID')

# net_GPU_0 = slim.conv2d(net_GPU_0, 128, [5, 5], stride=1, padding='VALID')
# net_GPU_1 = slim.conv2d(net_GPU_1, 128, [5, 5], stride=1, padding='VALID')

# net_GPU_0 = slim.max_pool2d(net_GPU_0, [3, 3], stride=2, padding='VALID')
# net_GPU_1 = slim.max_pool2d(net_GPU_1, [3, 3], stride=2, padding='VALID')

# net = tf.concat([net_GPU_0, net_GPU_1], 3)

# net_GPU_0 = slim.conv2d(net, 192, [3, 3], stride=1, padding='VALID')
# net_GPU_1 = slim.conv2d(net, 192, [3, 3], stride=1, padding='VALID')

# net_GPU_0 = slim.conv2d(net_GPU_0, 192, [3, 3], stride=1, padding='VALID')
# net_GPU_1 = slim.conv2d(net_GPU_1, 192, [3, 3], stride=1, padding='VALID')


# # 0327
# import tensorflow as tf
# import tensorflow.contrib.slim as slim

# p_l = tf.ones(shape=[10, 28, 28, 192], dtype=tf.float32)
# def inception(p_l):
#     l1_1 = slim.conv2d(p_l, 64, [1, 1], padding='SAME')    # 为通道信息的交换   # shape = [10, 28, 28, 64]
#     l3_3 = slim.conv2d(p_l, 128, [3, 3], padding='SAME')    # 特征提取    # shape = [10, 28, 28, 128]
#     l5_5 = slim.conv2d(p_l, 128, [5, 5], padding='SAME')    # 特征提取    # shape = [10, 28, 28, 128]
#     l_pool = slim.max_pool2d(p_l, [3, 3], stride=2, padding='SAME')    # shape = [10, 14, 14, 192]
#     l_pool = tf.pad(l_pool, [[0, 0], [7, 7], [7, 7], [0, 0]])     # *****
#     # 进行拼接
#     rst = tf.concat([l1_1, l3_3, l5_5, l_pool], axis=3)

#     return rst

# # res = inception(p_l)
# # print(res)


# # inception3
# p_l = tf.ones(shape=[10, 28, 28, 192], dtype=tf.float32)


# def inception_3(p_l):
#     l_1 = slim.conv2d(p_l, 64, [1, 1], padding='SAME')  # [10, 28, 28, 64]
#     l1_1 = slim.conv2d(p_l, 96, [1, 1], padding='SAME')  # [10, 28, 28, 96]
#     l2_3 = slim.conv2d(l1_1, 128, [3, 3], padding='SAME')  # [10, 28, 28, 128]
#     l1_11 = slim.conv2d(p_l, 16, [1, 1], padding='SAME')  # [10, 28, 28, 16]
#     l2_5 = slim.conv2d(l1_11, 32, [5, 5], padding='SAME')  # [10, 28, 28, 32]
#     l1_3 = slim.max_pool2d(p_l, [3, 3], stride=2, padding='SAME')  # [10, 14, 14, 192]
#     p2_1 = slim.conv2d(l1_3, 32, [1, 1], padding='SAME')  # [10, 14, 14, 32]

#     l_pool = tf.pad(p2_1, [[0, 0], [7, 7], [7, 7], [0, 0]])

#     rst = tf.concat([l_1, l2_3, l2_5, l_pool], axis=3)

#     # 打印模型参数数量
#     from functools import reduce
#     from operator import mul

#     def get_num_params():
#         num_params = 0
#         for variable in tf.trainable_variables():
#             print(variable)
#             shape = variable.get_shape()
#             num_params += reduce(mul, [dim.value for dim in shape])   # ****
#         return num_params

#     print('参数数量为：%d' %get_num_params())


#     return rst

# res = inception_3(p_l)
# print(res)


# # usual
# def usual(p_l):
#     l1 = slim.conv2d(p_l, 256, [3, 3], padding='SAME')

#     return l1

# res = usual(p_l)
# print(res)


# import tensorflow as tf
# import tensorflow.contrib.slim as slim


# p_l = tf.ones(shape=[10, 224, 224, 192], dtype=tf.float32)
# def inception(p_l):
#     l1_7 = slim.conv2d(p_l, 64, [3, 3], stride=2, padding='SAME')

#     l2_3 = slim.max_pool2d(l1_7, [3, 3], stride=2, padding='SAME')

#     l3_1 = slim.conv2d(l2_3, )


# 0328
# import tensorflow as tf
# import tensorflow.contrib.slim as slim

# def fully_con_BN(inputs, num_cell, is_traing=True):
#     res = slim.fully_connected(inputs, num_cell, activation_fn=None)

#     if is_traing:
#         mean, var = tf.nn.moments(inputs, axes=0)   # 均值、方差
#         ema = tf.train.ExponentialMovingAverage(decay=0.99)  # 滑动平均模型
#         ema_apply_op = ema.apply([mean, var])    # 利用滑动平均模型更新
#         with tf.control_dependencies([ema_apply_op]):
#             outputs = tf.nn.batch_normalization(inputs,
#                                         mean=ema.average(mean),
#                                         variance=ema.average(var),
#                                         offset=0,
#                                         scale=1,
#                                         variance_epsilon=0.001)
#     else:
#         outputs = tf.nn.batch_normalization(inputs,
#                                         mean=ema.average(mean),
#                                         variance=ema.average(var),
#                                         offset=0,
#                                         scale=1,
#                                         variance_epsilon=0.001)
#     outputs = tf.nn.relu(outputs)
    
#     return outputs


# print(fully_con_BN(tf.ones(shape=[128, 1024]), 512))


# # slim用法
# inputs = tf.ones(shape=[128, 1024])
# res = slim.fully_connected(inputs, 1024, activation_fn=None)
# res = slim.batch_norm(res, is_traing=True)

# ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(ops):
#     train_op = tf.train.AdadeltaOptimizer().minimize(loss)


# 0330
# import tensorflow as tf
# import tensorflow.contrib.slim as slim

# def res_unit(inputs, kernel_num):
#     net = slim.conv2d(inputs, kernel_num, [3, 3], padding='SAME')
#     net = slim.conv2d(net, kernel_num, [3, 3], padding='SAME')

#     net = tf.add(net, inputs)

#     return net


# def res_unit(inputs, last_layer_kernel_num, kernel_num, stride=1):
#     net = slim.conv2d(inputs, kernel_num, [3, 3], stride=stride, padding='SAME')
#     net = slim.conv2d(net, kernel_num, [3, 3], padding='SAME')

#     need_trans = False
#     if last_layer_kernel_num != kernel_num:
#         need_trans = True
#     if stride != 1:
#         need_trans = True
#     if need_trans is True:
#         inputs = slim.conv2d(inputs, kernel_num, [1, 1], padding='SAME', activation_fn=None)

#     net = tf.add(net, inputs)

#     return net


# import tensorflow as tf
# import tensorflow.contrib.slim as slim

# def conv2d(inputs, kernel_num, kernel_size, stride):
#     net = slim.conv2d(inputs, kernel_num, kernel_size, stride=stride, activation_fn=None, padding='SAME')
#     net = slim.batch_norm(net)   # 批标准化
#     return tf.nn.relu(net)

# # 定义的残差块
# def res_block(inputs, kernel_num, last_kernel_num, stride=1):
#     net = conv2d(inputs, kernel_num, [1, 1], stride=1)
#     net = conv2d(net, kernel_num, [3, 3], stride=stride)
#     net = conv2d(net, kernel_num, [1, 1], stride=1)

#     if stride != 1:
#         inputs = slim.max_pool2d(inputs, [3, 3], stride=2, padding='SAME')
#     if kernel_num != last_kernel_num:
#         tf.pad(inputs, [[0, 0], [0, 0], [0, 0], [int(last_kernel_num/2), last_kernel_num-int(last_kernel_num/2)]])

#     return tf.add(net, inputs)

# inputs = tf.ones(shape=[10, 224, 224, 3])
# kernel_num = 128
# stride = 2

# # tmp = res_block(inputs, kernel_num, 64, stride)

# conv1 = conv2d(inputs, 64, [7, 7], stride=2)
# conv2 = slim.max_pool2d(conv1, [3, 3], 2, padding='SAME')

# conv2_x = res_block(conv2, 64, 64, 1)
# conv2_x = res_block(conv2_x, 64, 64, 1)

# conv3_x = res_block(conv2_x, 128, 64, 2)
# conv3_x = res_block(conv3_x, 128, 128, 1)

# ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

# with tf.control_dependencies([tf.group(*ops)]):
#     train_op = tf.train.AdamOptimizer().minimize(loss)


# # 0404
# # 使用tensorflow实现IOU计算
# def calc_iou(box1, box2):
#     left_up = tf.maximum(box1[:2], box2[:2])
#     right_down = tf.minimum(box1[2:], box2[2:])
#     intersection = tf.maximum(0., right_down - left_up)
#     inter_square = intersection[0] * intersection[1]

#     box1_square = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     box2_square = (box2[2] - box2[0]) * (box2[3] - box2[1])

#     iou = inter_square / (box1_square + box2_square - inter_square + 1e-8)

#     return iou


# # with tf.Session() as sess:
# #     iou = calc_iou([10., 20., 60., 60.], [50., 40., 80., 90.])
# #     print(sess.run(iou))

# #     iou = calc_iou([10., 20., 40., 60.], [50., 40., 80., 90.])
# #     print(sess.run(iou))


# 0408  求取map
import  sklearn.metrics as metrics
a = metrics.average_precision_score(
    [0, 1, 0, 1, 0, 0, 1, 0],
    [0.2, 0.4, 0.1, 0.3, 0.6, 0.25, 0.9, 0.35]
)
b = metrics.average_precision_score(
    [0, 0, 0, 0, 1, 1, 0, 1],
    [0.1, 0.5, 0.15, 0.4, 0.3, 0.65, 0.05, 0.45]
)
c = metrics.average_precision_score(
    [1, 0, 1, 0, 0, 0, 0, 0],
    [0.7, 0.1, 0.75, 0.3, 0.1, 0.1, 0.05, 0.2]
)

# print((a+b+c)/3)

# nms
def nms(boxes, threshold=0.7):
    boxes.sort(key=lambda boxes:boxes[4], reverse=True)
    res_boxes = []
    while (len(boxes) != 0):
        res_boxes.append(boxes[0])
        del boxes[0]

        keep_num = 0
        for i in range(len(boxes)):
            iou = calc_iou(res_boxes[-1][0:4], boxes[keep_num][0:4])
            if iou > threshold:
                del boxes[keep_num]
            else:
                keep_num += 1

    return res_boxes

# boxes = [
#     [0, 0, 50, 50, 0.9],

# ]

import numpy as np
import tensorflow as tf


outputs = tf.random_uniform([2, 8], 0, 1, dtype=tf.float32)
labels = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0.2, 0.3, 0.3, 0.8, 0, 0, 1]], tf.float32)

labels_confidence = labels[:, 0:1]

# 对预测结果进行处理，使得label的confidence为0时：
# 对应的output，除confidence以外的值都为0
# 即找到所有label的confidence == 0的索引并使box, class预测值为0

trans = tf.cast(tf.equal(labels_confidence, 1), tf.float32)
mask = tf.concat([tf.ones_like(labels)[:, 0:1],
                    (tf.ones_like(labels) * trans)[:, 1:]], 1)
outputs = outputs * mask

# 均方误差代价函数
# cost = tf.reduce_mean(tf.square(labels - outputs))
# import random
# random.randint


# 0418
from nets import resnet_v2
def build_network(self, istraining):
    with slim.arg_score(resnet_v2.resnet_arg_scope(batch_arg_scope(batch_norm_decay=0.92))):
        self.image = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, 3])
        self.labels = tf.placeholder(
            tf.float32,
            [self.batch_size, self.cell_size, self.cell_size, 5 + self.classes]
        )
        net_, end_points = resnet_v2.resnet_v2_50(self.images, is_training=is_training)
        net = end_points['resnet_v2_v50/block4/unit_3/bottleneck_v2/con2d']

        if is_traing:
            self.res_saver = tf.train.Saver()

        
        net = slim.conv2d(net, 120, [1, 1], padding='SAME', activation_fn=None)
        net = slim.batch_norm(net, activation_fn=tf.nn.relu, is_training=is_training)
        logits = slim.conv2d(net, 30, [2, 2], stride=2, padding='SAME', activation_fn=None)
        self.saver = tf.train.Saver()

    return logits
