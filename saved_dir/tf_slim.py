import tensorflow.contrib.slim as slim
import tensorflow as tf
# variable
weights = slim.variable('weight', shape=[10, 10, 3, 3], 
                                        initializer=tf.truncated_normal_initializer(stddev=0.1), 
                                        regularizer=slim.l2_regularizer(0.05),
                                        device='/CPU:0')

# model variable
weights = slim.model_variable('weights', shape=[10, 10, 3, 3],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1), 
                                        regularizer=slim.l2_regularizer(0.05),
                                        device='/CPU:0')
model_variables = slim.get_model_variables()
 
# regular variables
my_var = slim.variable('my_var', shape=[20, 1], 
                                initializer=tf.zeros_initializer())
regular_variables_and_model_variables = slim.get_variables()

## 添加模型参数
my_model_variable = CreateViaCustomCode()
#letting tf-slim know about the additional variable
slim.add_model_variable(my_model_variable)


input = ...
net = slim.conv2d(input, 128, [3, 3], scope='conv1_1')


net = ...
net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')
net = slim.max_pool2d(net, [2, 2], scope='pool2')
# ->for loop
net = ...
for i in range(3):
    net = slim.conv2d(net, 256, [3, 3], scope='conv3_%d'%(i+1))
net = slim.max_pool2d(net, [2, 2], scope='pool2')
# ->repeat
net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
net = slim.max_pool2d(net, [2, 2], scope='pool2')



# verbose way
x = slim.fully_connected(x, 32, scope='fc/fc_1')
x = slim.fully_connected(x, 64, scope='fc/fc_2')
x = slim.fully_connected(x, 128, scope='fc/fc_3')
# ->slim.stack
slim.stack(x, slim.fully_connected, [32, 64, 128], scope='fc')

# verbose way:
x = slim.conv2d(x, 32, [3, 3], scope='core/core_1')
x = slim.conv2d(x, 32, [1, 1], scope='core/core_2')
x = slim.conv2d(x, 64, [3, 3], scope='core/core_3')
x = slim.conv2d(x, 64, [1, 1], scope='core/core_4')
# ->using stack
slim.stack(x, slim.conv2d, [(32, [3, 3]), (32, [1, 1]), (64, [3, 3]), (64, [1, 1])], scope='core')


# scopes
net = slim.conv2d(inputs, 64, [11, 11], 4, padding='SAME'),
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                weights_regularizer=slim.l2_regularizer(0.0005), scope='conv1')
net = slim.conv2d(net, 128, [11, 11], padding='VALID',
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    weights_regularizer=slim.l2_regularizer(0.0005), scope='conv2')
                    weights_regularizer=slim.l2_regularizer(0.0005), scope='conv2')
net = slim.conv2d(net, 256, [11, 11], padding='SAME', 
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    weights_regularizer=slim.l2_regularizer(0.0005), scope='conv3')
# ->easier
padding = 'SAME'
initializer = tf.truncated_normal_initializer(stddev=0.01)
regularizer = slim.l2_regularizer(0.0005)
net = slim.conv2d(inputs, 64, [11, 11], 4, 
                    padding=padding,
                    weights_initializer=initializer,
                    weights_regularizer=regularizer,
                    scope='conv1')
# ->easiest
with slim.arg_scop([slim.conv2d], padding='SAME', 
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.conv2d(inputs, 64, [11, 11], scope='conv1')
    net = slim.conv2d(net, 128, [11, 11], padding='VALID', scope='conv2')
    net = slim.conv2d(net, 256, [11, 11], scope='conv3')

# 嵌套
with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    weights_regularizer=slim.l2_regularizer(0.0005)):
    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
        net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
        net = slim.conv2d(net, 256, [5, 5],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.03),
                            scope='conv2')
        net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc') 

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     w = sess.run(model_variables)
#     print(w)

# for example      VGG16
def vgg16(inputs):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3,3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        net = slim.fully_connected(net, 4096, scope='fc6')
        net = slim.dropout(net, 0.5, scope='dropout6')
        net = slim,fully_connected(net, 4096, scope='fc7')
        net = slim.dropout(net, 0.5, scope='dropout7')
        net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
    return net


# Training Models： 训练Tensorflow模型需要一个模型，一个损失函数，梯度计算和一个训练例程，迭代地计算
# 相对于损失的模型权重的梯度并相应地更新权重。TF-Slim提供了常用的损失函数和一组运行训练和评估例程
# 的辅助函数。

# Losses
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
vgg = nets.vgg

# Load the images and labels
images, labels = ...

# Create the model
predictions, _ = vgg.vgg_16(images)

# Define the loss functions and get the total loss.
loss = slim.losses.softmax_cross_entropy(predictions, labels)


# 多任务模型产生多个输出
# Load the images and labels
images, scene_labels, depth_labels = ...
# Create the model
scene_predictions, depth_predictions = CreateMultiTaskModel(images)
# Define the loss functions and get the total loss
classfication_loss = slim.losses.softmax_cross_entropy(sene_predictions, scene_labels)
sum_of_squares_loss = slim.losses.sum_of_squares(depth_predictions, DeprecationWarningth_labels)
# The following two lines have the same effect:
total_loss = classfication_loss + sum_of_squares_loss
total_loss = slim.losses.get_total_loss(add_regularization=False)


# 添加自定义的损失函数
# Load the imagees and labels
images, scene_labels, depth_labels, pose_labels = ...
# Create the model
scene_predictions, depth_predictions, pose_predictions = CreateMultiTaskModel(images)
# Define the loss functions and get the total loss
classification_loss = slim.losses.softmax_cross_entropy(scene_predictions, scene_labels)
sum_of_squares_loss = slim.losses.sum_of_squares(depth_predictions, depth_labels)
pose_loss = MyCustomLossFunction(pose_predictions, pose_labels)
slim.losses.add_loss(poss_loss)  # letting TF-Slim know about the additional loss

# The following two ways to compute the total loss are equivalent:
regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
# 正则项也是需要手动添加进loss当中的，不然最后计算的时候就不优化正则目标了 *****
total_loss1 =  classification_loss + sum_of_squares_loss + pose_loss + regularization_loss

# Regularization loss is included in the total loss by default
total_loss2 = slim.losses.get_total_loss()



# Training Loop:一旦我们指定了模型，损失函数和优化方案，我们可以调用
# slim.learning.create_train_op 并 slim.learning.train执行优化：
g = tf.Graph()
# Create the model and specify the losses...
# ...
total_loss = slim.losses.get_total_loss()
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# create_train_op ensures that each time we ask for the loss, the update_ops
# are run and the gradients being computed are appled too.
train_op = slim.learning.create_train_op(total_loss, optimizer)
logdir = ....  # where checkpoints are stored

slim.learning.train(
    train_op,
    logdir,
    number_of_steps=1000,
    save_summaries_secs=300,   # 每5分钟进行摘要
    save_interval_secs=600     # 每10分钟保存以此模型检查点
)


# Working Example:Training the vgg16
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets

slim = tf.contrib.slim
vgg = nets.vgg
...
train_log_dir = ...
if not tf.gfile.Exists(train_log_dir):
    tf.gfile.MakeDirs(train_log_dir)

with tf.Graph().as_default():
    # set up the data loading:
    images, labels = ...
    # Define the model:
    predictions = vgg.vgg_16(images, is_training=True)   # is_training表此模型是否在训练
    # Specify the loss function:
    slim.losses.softmax_cross_entropy(predictions, labels)

    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('lossses/total_loss', total_loss)
    # Specify the optimization schee:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)
    # create_train_op that ensures that when we evaluate it to get the loss,
    # the update_ops are done and the gradient updates are computed
    train_tensor = slim.learning.create_train_op(total_loss, optimizer)
    # Actually runs training
    slim.learning.train(train_tensor, train_log_dir)


# Fine-Tuning Existing Models (微调现有模型)
# Create some variables
v1 = tf.Variable(..., name='v1')
v2 = tf.Variable(..., name='v2')
...
# Add ops to restore all the variables
restorer = tf.train.Saver()
# Add ops to restore some variables
restorer = tf.train.Saver()
# Later, launch the model, use the saver to restore variables from disk, and 
# do some work with model
with tf.Session() as sess:
    # Restore variables from disk.
    restorer.restore(sess, '/tmp/model.ckpt')
    print('Model restored')
    # Do soome work with the model
    ...



# Partially Restoring Models(部分恢复模型)
# Create some variables
v1 = slim.variable(name='v1', ...)
v2 = slim.variable(name='nested/v2', ...)
...
# Get list of variables to restore(which contains only 'v2').These are all
# equivalent methods:
variables_to_restore = slim.get_variables_by_name('v2')
# or
variables_to_restore = slim.get_variables_by_suffix('2')   # suffix 后缀
# or
variables_to_restore = slim.get_variables(scope='nested')
# or
variables_to_restore = slim.get_variables_to_restore(include=['nested'])
# or
variables_to_restore = slim.get_variables_to_restore(exclude=['v1'])
# Create the saver which will be used to restore the variables
restorer = tf.train.Saver(variables_to_restor)

with tf.Session() as sess:
    # Restore variables from disk
    restorer.restore(sess, '/tmp/model.ckpt')
    print('Model restored')
    # Do some work with the model
    ...


# Restoring models with different variable names(使用不同的变量名称恢复模型)
# 当从checkpoint加载变量时，Saver先在checkpoint定位变量名，然后映射到当前图的变量中。我们也可以
# 通过向saver传递一个变量列表来创建saver。这时，在checkpoint文件中用于定位的变量名可以隐式地从各自
# 的.op.name中获得。
# 当checkpoint文件中的变量名与当前图中的变量名完全匹配时，这会运行得很好。但是，有时我们想从一个变量名
# 与当前图中的变量名不同的checkpoint文件中装载一个模型。这时，我们必须提供一个saver字典，这个字典对checkpoint
# 中的每个变量和每个图变量进行了一一映射
# Assuming than 'conv1/weights' should be restored from 'vgg16/conv1/weights'
def name_in_checkpoint(var):
    return 'vgg16/' + var.op.name
# Assuming than 'conv1/weights' and 'conv1/bias' should be restored from 'conv1/params1'
# and 'conv1/params2'
def name_in_checkpoint(var):
    if 'weights' in var.op.name:
        return var.op.name.replace('weights', 'params1')
    if 'bias' in var.op.name:
        return var.op.name.replace('bias', 'params2')

variables_to_restore = slim.get_model_variables()
variables_to_restore = {name_in_checkpoint(var):var for var in variables_to_restore}
restorer = tf.train.Saver(variables_to_restore)

with tf.Session() as sess:
    # restore variables from disk
    restorer.restore(sess, '/tmp/model.ckpt')



# Fine-Tuning a Model on a different task(在不同的任务上对模型对模型进行微调)
# 考虑一下我们有一个预先训练的vgg16，该模型在ImageNet数据集上进行了训练，该数据集有1000类。但是，
# 我们希望将其应用于仅有20个类别的Pascal VOC 数据集。为此，我们可以使用不包括最后一层的预先训练的模型的值
# 来初始化我们的新模型
# Load the Pascal VOC data
image, label = MyPascalVocDataLoader(...)
iamges, labels = tf.train.batch([image, label], batch_size=32)
# Create the model
predictions = vgg.vgg_16(images)
train_op = slim.learnng.create_train_op(...)
# Specify where the Model, trained on ImageNet, was saved
model_path = '/path/to/pre_trained_on_imagenet.chekpoint'
# Specify where the new model will live:
log_dir = '/path/to/my_pascal_model_dir/'
# Restore only the convolutional layers:
variables_to_restore = slim.get_variables_to_restore(exclude=['fc6', 'fc7', 'fc8'])
init_fn = assign_from_checkpoint_fn(model_path, variables_to_restore)
# start training
slim.learning.train(train_op, log_dir, init_fn=init_fn)




# Evaluating Models(评估模型)
images, labels = LoadTestData(...)
predictions = MyModel(images)
mae_value_op, mae_update_op = slim.metrics.streaming_mean_absolute_error(predictions, labels)
mre_value_op, mre_update_op = slim.metrics.streaming_mean_relative_error(predictions, labels)
pl_value_op, pl_ipdate_op = slim.metrics.precentage_less(mean_relative_errors, 0.3)
# As the example illustrates,the creation of a metric returns two values:a value_op and an update_op.
# The value_op is an idempotent that returns the value of metric.The update_op is an operation that performs
# the aggregation step mentioned above as well as returning the value of the metric.
# 如示例所示，度量的创建返回两个值：一个value_op和update_op。value_op是一个幂等操作，返回度量的当前值。
# update_op是执行上述聚合步骤以及返回度量值的操作

# Keeping track of each value_op and update_op can be laborious(费力). To deal with this,TF-Slim
# provides two convenience functions:
# Aggregates the value and update ops in two lists:
value_ops, update_ops = slim.metrics.aggregate_metrics(
    slim.metrics.streaming_mean_absolute_error(predictions, labels),
    slim.metrics.streaming_mean_squared_error(predictions, labels)
)
# Aggregates the value and update ops in two dictionaries:
names_to_values, names_to_updates = slim.metrics.aggregate_metricmap(
    {
        'eval/mean_absolute_error':slim.metrics.streaming_mean_absolute_error(predictions, labels),
        'eval/mean_squared_error':slim.metrics.streaming_mean_squared_error(predictions, labels),
    }
)


# Working example
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets

slim = tf.contrib.slim
vgg = nets.vgg
# Load the data
images, labels = load_data(...)
# Define the network
predictions = vgg.vgg_16(images)
# Choose the metrics to compute:
names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(
    {
        'eval/mean_absolute_error':slim.metrics.streaming_mean_absolute_error(predictions, labels),
        'eval/mean_squared_error':slim.metrics.streaming_mean_squared_error(predictions, labels),
    }
)
# Evaluate the model using 1000 batches of data:
num_batches = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variavles_initializer())

    for batch_id in range(num_batches):
        sess.run(names_to_updates.values())
    
    metric_values = sess.run(names_to_values.values())
    for metric, value in zip(names_to_values.key(), metric_values):
        print('Metric %s has value:%f' % (metric, value))


# Evaluation Loop
import tensorflow as tf
slim = tf.contrib.slim
# Load the data
images, labels = load_data(...)
# Define the network
predictions = MyModel(images)
# Choose the metrics to compute:
names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(
    {
        'accuracy':slim.metrics.accuracy(predictions, labels),
        'precision':slim.metrics.precision(predictions, labels),
        'recall':slim.metrics.recall(mean_relative_errors, 0.3),   # 召回率
    }
)
# Create the summary ops such that they also print out to std output:
summary_ops = []
for metric_name, metric_value in names_to_values.iteritems():
    op = tf.summary.scalar(metric_name, metric_value)
    op = tf.Print(op, [metric_value], metric_name)
    summary_ops.append(op)
num_examples = 10000
batch_size = 32
num_batches = math.ceil(num_examples / float(batch_size))
# Setup the global step
slim.get_or_create_global_step()
output_dir = ... # where the summaries are stored
eval_interval_secs = ...  # How often to run the evaluation
slim.evaluation.evaluation.evaluation_loop(
    'local',
    chckpoint_dir,
    log_dir,
    num_evals=num_batches,
    eval_op=names_to_uodates.values(),
    summary_op=tf.summary.merge(summary_ops),
    eval_interval_secs=eval_interval_secs
)

