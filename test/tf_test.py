# -*- coding: utf-8 -*-  
'''
tensorflow 测试

@author: luoyi
Created on 2021年1月1日
'''
import tensorflow as tf
import numpy as np

#    print时不用科学计数法表示
np.set_printoptions(suppress=True)  


#    custom training
class TestModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(TestModel, self).__init__(name='TestModel', **kwargs)
        self.build_layer()
        pass
    def build_layer(self):
        self.fc1 = tf.keras.models.Sequential([
                tf.keras.layers.Dense(name='fc11', units=4, activation=tf.keras.activations.tanh, kernel_initializer=tf.initializers.he_normal(), bias_regularizer=tf.initializers.zeros()),
                tf.keras.layers.Dense(name='fc12', units=16, activation=tf.keras.activations.tanh, kernel_initializer=tf.initializers.he_normal(), bias_regularizer=tf.initializers.zeros()),
                tf.keras.layers.Dense(name='fc13', units=16, activation=tf.keras.activations.tanh, kernel_initializer=tf.initializers.he_normal(), bias_regularizer=tf.initializers.zeros()),
                tf.keras.layers.Dense(name='fc14', units=4, activation=tf.keras.activations.tanh, kernel_initializer=tf.initializers.he_normal(), bias_regularizer=tf.initializers.zeros())
            ])
#         self.fc2 = tf.keras.models.Sequential([
#                 tf.keras.layers.Dense(name='fc21', units=64, activation=tf.keras.activations.relu, kernel_initializer=tf.initializers.he_normal(), bias_regularizer=tf.initializers.zeros()),
#                 tf.keras.layers.Dense(name='fc22', units=32, activation=tf.keras.activations.relu, kernel_initializer=tf.initializers.he_normal(), bias_regularizer=tf.initializers.zeros()),
#                 tf.keras.layers.Dense(name='fc23', units=16, activation=tf.keras.activations.relu, kernel_initializer=tf.initializers.he_normal(), bias_regularizer=tf.initializers.zeros()),
#                 tf.keras.layers.Dense(name='fc24', units=8, activation=tf.keras.activations.relu, kernel_initializer=tf.initializers.he_normal(), bias_regularizer=tf.initializers.zeros())
#             ])
        self.out = tf.keras.layers.Dense(1)
        pass
    def call(self, x, training=None, mask=None):
        y = self.fc1(x)
#         y = self.fc2(y)
        y = self.out(y)
        return y
    pass


epochs = 100
batch_size = 16
train_num = 20000
val_num = 100

#    学习函数y = x² + 1
def f(x):
    return tf.pow(x, 1) + 1.
#    数据集
x_train = tf.linspace(start=-1, stop=1, num=train_num)
x_train = tf.expand_dims(x_train, axis=-1)
y_train = f(x_train)
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
x_val = tf.linspace(start=-1, stop=1, num=100)
x_val = tf.expand_dims(x_val, axis=-1)
y_val = f(x_val)
db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

#    损失，优化器，模型定义
loss_fun = tf.losses.MSE
optimizers = tf.optimizers.Adam(learning_rate=0.001)
model = TestModel()
model.compile(optimizer=optimizers, 
              loss=loss_fun, 
              metrics=[tf.metrics.mean_absolute_error])
print('model.optimizer:', model.optimizer)
print('model.loss:', model.loss)
print('model.metrics', model.metrics)


#    各种记录指标
train_loss = tf.metrics.Mean('train_loss', dtype=tf.float32)
train_metric = tf.metrics.Mean('train_metric', dtype=tf.float32)
val_loss = tf.metrics.Mean('val_loss', dtype=tf.float32)
val_metric = tf.metrics.Mean('val_metric', dtype=tf.float32)


train_step_signature = [
        tf.TensorSpec(shape=(None,1), dtype=tf.float64),
        tf.TensorSpec(shape=(None,1), dtype=tf.float64),
        tf.TensorSpec(shape=(None), dtype=tf.int64)
    ]
#    单步训练过程
@tf.function(input_signature=train_step_signature)
def train_step(x, y, step=0):
    with tf.GradientTape() as tape:
        y_prod = model(x, training=True)
        l = model.loss(y, y_prod)
        pass
    grads = tf.gradients(l, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(l)
    train_metric(l)
    tf.summary.scalar('train_loss', tf.reduce_mean(l), step=step)
    tf.summary.scalar('train_metric', tf.reduce_mean(l), step=step)
    return l
@tf.function(input_signature=train_step_signature)
def val_step(x, y, step=0):
    y_prod = model(x)
    l = loss_fun(y, y_prod)
    val_loss(l)
    val_metric(l)
    tf.summary.scalar('val_loss', tf.reduce_mean(l), step=step)
    tf.summary.scalar('val_metric', tf.reduce_mean(l), step=step)
    return l


tensorboard_dir = '/Users/irenebritney/Desktop/workspace/eclipse-workspace2/faster_rcnn/logs/tf_test'
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir='/Users/irenebritney/Desktop/workspace/eclipse-workspace2/faster_rcnn/logs/tf_test',               #    tensorboard主目录
                                                                                 histogram_freq=1,                      #    对于模型中各个层计算激活值和模型权重直方图的频率（训练轮数中）。 
                                                                                                                        #        如果设置成 0 ，直方图不会被计算。对于直方图可视化的验证数据（或分离数据）一定要明确的指出。
                                                                                 write_graph=True,                      #    是否在 TensorBoard 中可视化图像。 如果 write_graph 被设置为 True
                                                                                 write_grads=True,                      #    是否在 TensorBoard 中可视化梯度值直方图。 
                                                                                                                        #        histogram_freq 必须要大于 0
                                                                                 batch_size=batch_size,                 #    用以直方图计算的传入神经元网络输入批的大小
                                                                                 write_images=True,                     #    是否在 TensorBoard 中将模型权重以图片可视化，如果设置为True，日志文件会变得非常大
                                                                                 embeddings_freq=None,                  #    被选中的嵌入层会被保存的频率（在训练轮中）
                                                                                 embeddings_layer_names=None,           #    一个列表，会被监测层的名字。 如果是 None 或空列表，那么所有的嵌入层都会被监测。
                                                                                 embeddings_metadata=None,              #    一个字典，对应层的名字到保存有这个嵌入层元数据文件的名字
                                                                                 embeddings_data=None,                  #    要嵌入在 embeddings_layer_names 指定的层的数据。 Numpy 数组（如果模型有单个输入）或 Numpy 数组列表（如果模型有多个输入）
                                                                                 update_freq='batch'                    #    'batch' 或 'epoch' 或 整数。
                                                                                                                        #        当使用 'batch' 时，在每个 batch 之后将损失和评估值写入到 TensorBoard 中。
                                                                                                                        #        同样的情况应用到 'epoch' 中。
                                                                                                                        #        如果使用整数，例如 10000，这个回调会在每 10000 个样本之后将损失和评估值写入到 TensorBoard 中。注意，频繁地写入到 TensorBoard 会减缓你的训练。
                                                         )
reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                        factor=0.1,             #    每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
                                                                        patience=1,             #    当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
                                                                        mode='auto',            #    ‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少
                                                                        epsilon=0.000000001,    #    阈值，用来确定是否进入检测值的“平原区” 
                                                                        cooldown=0,             #    学习率减少后，会经过cooldown个epoch才重新进行正常操作
                                                                        min_lr=0                #    学习率的下限（下不封顶）
                                                                        )

train_summary_writer = tf.summary.create_file_writer(tensorboard_dir + "/train")
val_summary_writer = tf.summary.create_file_writer(tensorboard_dir + '/validation')
callbacks = tf.keras.callbacks.CallbackList(callbacks=[
                                                tensorboard_cb,
                                                reduce_lr_on_plateau
                                            ],
                                            add_history=True,
                                            add_progbar=1,
                                            verbose=1,
                                            epochs=epochs,
                                            model=model,
                                            steps=train_num / batch_size)
# callbacks,
#             add_history=True,
#             add_progbar=verbose != 0,
#             model=self,
#             verbose=verbose,
#             epochs=epochs,
#             steps=data_handler.inferred_steps

prev_val_loss = 0.
down_lr_times = 10
num_lr_times = 0
logs = {}
#    训练之前回调
callbacks.on_train_begin(logs)
all_train_step = 0
all_val_step = 0
#    训练过程
for epoch in range(epochs):
    #    迭代训练集
    batch = 0
    #    每个epoch开始回调
    callbacks.on_epoch_begin(epoch)
    
    for x, y in db_train:
        callbacks.on_train_batch_begin(batch, logs)
        
        l = train_step(x, y, step=all_train_step)
        logs['train_loss'] = tf.reduce_mean(l)
        logs['train_lr'] = model.optimizer.lr
        all_train_step += 1
        batch += 1
        
        callbacks.on_train_batch_end(batch, logs)
        pass
    
    #    跑完训练集后跑验证集
    #    每个验证开始回调
    callbacks.on_test_begin(logs)
    batch = 0
    for x, y, in db_val:
        #    每个验证batch开始回调
        callbacks.on_test_batch_begin(batch, logs)
        batch += 1
        l = val_step(x, y, step=all_val_step)
        logs['val_loss'] = tf.reduce_mean(l)
        all_val_step += 1
        #    每个验证batch结束回调
        callbacks.on_test_batch_end(batch, logs)
        pass
    #    每个验证结束回调
    callbacks.on_test_end(logs)
    
#     print('epoch:{} lr:{} train_loss:{} val_loss:{}'.format(epoch, model.optimizer.lr, train_loss.result(), val_loss.result()))
    
    train_loss.reset_states()
    val_loss.reset_states()
    train_metric.reset_states()
    val_metric.reset_states()
    
    #    每个epoch结束回调
    callbacks.on_epoch_end(epoch, logs)
    pass
#    训练完成回调
callbacks.on_train_end(logs)

x_test = tf.linspace(-1, 1, 10)
print(x_test)
x_test = tf.expand_dims(x_test, axis=-1)
print(model(x_test))

